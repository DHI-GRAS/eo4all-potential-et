#!/usr/bin/env python

import datetime
from dateutil import rrule
import uuid
from dataclasses import asdict
from functools import partial
import json

from kafka.env_configuration import SELECTED_CONFIG
from kafka.consumer import KafkaConsumerThread
from kafka.producer import KafkaProducer
from kafka.models import Message, custom_asdict_factory

import potential_et


def register_consumer(name, topic, broker_address, handle_json_message_data=True,
                      run_as_separate_thread=False, consumer_group=str(uuid.uuid1())[0:10]):
    consumer = KafkaConsumerThread(topic, broker_address, handle_json_message_data,
                                   run_as_separate_thread, consumer_group=consumer_group)

    print(f"register {name} consumer with consumer group {str(consumer.consumer_group)}")

    # add a method "consume message" for consuming message
    consumer.start_consumer(partial(consume_message, producer_topic=topic))


def consume_message(request_str, producer_topic):
    message = Message.from_json(request_str)
    process_message(message)


def produce_message(message: Message, producer_topic):

    producer = KafkaProducer(broker_address=SELECTED_CONFIG.KAFKA_URI)
    dict_message = asdict(message, dict_factory=custom_asdict_factory)

    # print('#############')
    # print(dict_message)

    # publish message
    producer.publish_message(topic=producer_topic, message=dict_message)


def process_message(message: Message):
    # do something
    print(f"message with id {message.message_id} received")

    out_file_paths = []
    for tile in message.input_tiles:

        start_date = datetime.datetime.strptime(message.processing_start_date, "%Y%m%d")
        end_date = datetime.datetime.strptime(message.processing_end_date, "%Y%m%d")

        for date in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date, interval=10):
            json_string = json.dumps({"aoi_name": tile,
                                      "date": date.strftime("%Y-%m-%d"),
                                      "spatial_res": "s2",
                                      "temporal_res": "dekadal"})
            out_file_path = potential_et.run(json_string)
            out_file_paths.append(out_file_path)

    # Update parameters
    message.outputs = out_file_paths if out_file_paths else ["Data not available"]
    # message.output_styles=["Add here the list of styles (path) "]
    message.is_process_success = True

    # Send the message that the process was performed
    produce_message(message, SELECTED_CONFIG.TOPIC_SERVICE)
    print(f"Sending message to topic: {SELECTED_CONFIG.TOPIC_SERVICE}.")


def main():

    # register consumer on topic using kafka uri and consumer group
    register_consumer(name="PotentialEtConsumer",
                      topic=SELECTED_CONFIG.TOPIC_ORDER,
                      broker_address=SELECTED_CONFIG.KAFKA_URI,
                      consumer_group=SELECTED_CONFIG.CONSUMER_GROUP)


main()
