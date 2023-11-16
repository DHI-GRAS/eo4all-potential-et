import logging

from kafka.consumer import KafkaConsumerThread


def handle_json_message_data(msg_value):
    logging.info(msg_value)


KafkaConsumerThread(
    topic="myTopic",
    broker_address='127.0.1.1',
    handle_json_message_data=True,
    run_as_separate_thread=False
).start_consumer(handle_json_message_data)
