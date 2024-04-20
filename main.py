#!/usr/bin/env python

import calendar
import datetime
import json
import uuid
import logging

from dataclasses import asdict
from dateutil import rrule
from functools import partial

import potential_et

from kafka.env_configuration import SELECTED_CONFIG
from kafka.consumer import KafkaConsumerThread
from kafka.producer import KafkaProducer
from kafka.models import Message, custom_asdict_factory


def register_consumer(name, topic, broker_address, handle_json_message_data=True,
                      run_as_separate_thread=False, consumer_group="potential-evapotranspiration"):
    consumer = KafkaConsumerThread(topic, broker_address, handle_json_message_data,
                                   run_as_separate_thread, consumer_group=consumer_group)

    print(f"register {name} consumer with consumer group {str(consumer.consumer_group)}")

    # add a method "consume message" for consuming message
    consumer.start_consumer(partial(consume_message, producer_topic=topic))


def consume_message(request_str, producer_topic):
    message = Message.from_dict(request_str)
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
    all_succeeded = True
    for tile in message.input_tiles:

        start_date = datetime.datetime.strptime(message.processing_start_date, "%Y%m%d")
        start_date = start_date.replace(day=1)
        end_date = datetime.datetime.strptime(message.processing_end_date, "%Y%m%d")
        end_date = end_date.replace(day=calendar.monthrange(end_date.year, end_date.month)[1])

        for date in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date, interval=10):
            if date.day > 21:
                continue
            json_string = json.dumps({"aoi_name": tile,
                                      "date": date.strftime("%Y-%m-%d"),
                                      "spatial_res": "s2",
                                      "temporal_res": "dekadal"})
            try:
                out_file_path = potential_et.run(json_string)
                if out_file_path:
                    out_file_paths.append(out_file_path)
            except Exception as e:
                # Flag the process as failed and stop processing the rest of the tiles
                logging.exception(f"Potential ET process failed. Parameters: {json_string}.")
                all_succeeded = False
                break

        if all_succeeded is False:
            break


    # Update parameters
    message.outputs = out_file_paths if out_file_paths else ["Data not available"]
    # message.output_styles=["Add here the list of styles (path) "]
    message.is_process_success = all_succeeded

    # Send the message that the process was performed
    produce_message(message, SELECTED_CONFIG.TOPIC_SERVICE)
    print(f"Sending message to topic: {SELECTED_CONFIG.TOPIC_SERVICE}.")


def main():

    # register consumer on topic using kafka uri and consumer group
    register_consumer(name="PotentialEtConsumer",
                      topic=SELECTED_CONFIG.TOPIC_ORDER,
                      broker_address=SELECTED_CONFIG.KAFKA_URI,
                      consumer_group=SELECTED_CONFIG.CONSUMER_GROUP)


if __name__ == "__main__":
    main()
