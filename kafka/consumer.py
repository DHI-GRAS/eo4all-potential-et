import json
import logging
import uuid

from concurrent.futures import ThreadPoolExecutor
from confluent_kafka import Consumer, KafkaError


class KafkaConsumerThread:
    """
    Start a new Consumer Thread for consuming and processing
    Kafka based messages
    """

    def __init__(self,
                 topic,
                 broker_address,
                 handle_json_message_data=True,
                 run_as_separate_thread=True,
                 consumer_group=str(uuid.uuid1())[0:10]
                 ):

        """
        Init the main class
        :param topic: the topic you want to subscribe to
        :param broker_address: the broker address
        :param handle_json_message_data: True if you want to handle json formatted data, False otherwise
        """
        self.topic = topic
        self.broker_address = broker_address
        self.handle_json_message_data = handle_json_message_data
        self.run_as_separate_thread = run_as_separate_thread
        self.consumer_group = consumer_group

    def start_consumer(self, handle_message_function):
        """
        Create the consumer, start it and begin consuming messages running a new thread
        :param handle_message_function: the function to handle consumed message
        """

        number_of_consumers=1
        with ThreadPoolExecutor(max_workers=number_of_consumers) as executor:

            for i in range(number_of_consumers):
                #feel free to setup the name you want
                complete_name = "consumer" + "_" + str(i)
                logging.info(f"start {complete_name} with consumer group {str(self.consumer_group)}")
                executor.submit(self.run_thread, handle_message_function)

    def run_thread(self, handle_message_value_function):
        """
        Run the thread and start processing incoming messages
        :param handle_message_value_function: the function to handle consumed message
        """

        consumer = Consumer({
            'bootstrap.servers': self.broker_address,
            'auto.offset.reset': 'latest',
            'group.id': self.consumer_group,
        })

        consumer.subscribe([self.topic])

        while True:
            try:
                msg = consumer.poll(1.0)

                if msg is None:
                    continue
                if msg.error():
                    logging.error("Consumer error: %s", msg.error())
                    continue
                value = msg.value()
                if self.handle_json_message_data:
                    value = json.loads(value.decode('utf8'))
                handle_message_value_function(value)
            except KafkaError as ke:
                logging.error(ke)
            except Exception as e:
                logging.error(e)
