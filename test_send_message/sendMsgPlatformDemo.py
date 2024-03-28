import uuid
from dataclasses import asdict
from functools import partial

from kafka.env_configuration import SELECTED_CONFIG
from kafka.consumer import KafkaConsumerThread
from kafka.producer import KafkaProducer
from kafka.models import Message, custom_asdict_factory
from kafka.utils import get_resources_path


def main():
    topic = SELECTED_CONFIG.TOPIC_ORDER

    # create a producer using kafka uri
    producer = KafkaProducer(broker_address=SELECTED_CONFIG.KAFKA_URI)

    # generate message_id unique for every send of the same message
    message = Message(message_id=str(uuid.uuid4()))

    message = {
        "message_id": "aaa374bb-f5d7-4158-aa6b-c415f005eeb9",
        "process_id": "823d36be-55d5-435e-a2b4-43d94d8107b9",
        "processor": "Potential Evapotranspiration mapping",
        "processor_type": "crop",
        "is_external_docker_processing": False,
        "internal_input_folder_path": "/mnt/volumes/adb_workspace/folder_823d36be-55d5-435e-a2b4-43d94d8107b9_Potential-Evapotranspiration_mapping/input",
        "internal_output_folder_path": "/mnt/volumes/adb_workspace/folder_823d36be-55d5-435e-a2b4-43d94d8107b9_Potential-Evapotranspiration_mapping/output",
        "external_folder_path": "https://eo4all-services.planetek.it/adb_workspace/folder_823d36be-55d5-435e-a2b4-43d94d8107b9_Potential-Evapotranspiration_mapping/",
        "request_timestamp": 1700825777.365063,
        "status": "start processing",
        "user_email": "dippolito@planetek.it",
        "processing_date": "20220801",
        "result_filename": "result",
        "periodicity": None,
        "processing_start_date": "20220801",
        "processing_end_date": "20220831",
        "result_configuration_path": "/mnt/volumes/adb_workspace/folder_823d36be-55d5-435e-a2b4-43d94d8107b9_Potential-Evapotranspiration_mapping/output/result_configuration.json",
        "message": None,
        "extended_perms": '{"users": {"dippolito": ["delete_resourcebase", "change_resourcebase_permissions", "publish_resourcebase", "change_resourcebase", "change_resourcebase_metadata", "download_resourcebase", "view_resourcebase"]}, "groups": {"contributors": ["download_resourcebase", "view_resourcebase"]}}',
        "compact_perms": '{"users": [{"id": 1001, "permissions": "manage"}], "groups": [{"id": 1, "permissions": "download"}], "organizations": []}',
        "create_map_with_layers": False,
        "optional_fields": {
            "folder_name": "folder_823d36be-55d5-435e-a2b4-43d94d8107b9_Potential-Evapotranspiration_mapping"
        },
        "map_creation_info": None,
        "is_process_success": True,
        "input_tiles": ["T48MYT", "T48MZT"],
        "outputs": None,
        "output_styles": None,
        "origin": "Potential-Evapotranspiration",
        "processing_tiestimation_minutes": 30,
        "username": "dippolito",
        "is_user_staff_or_admin": True,
    }

    dict_message = message

    print("Platform emulator to send a request to a process")
    print(f"Sending message to topic: {topic}. Message: {str(dict_message)}")

    # publish message
    producer.publish_message(topic=topic, message=dict_message)


main()
