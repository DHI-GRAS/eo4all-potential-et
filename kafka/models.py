from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import List


class ProcessingStatusEnum(Enum):
    SEND_REQUEST_TO_PROCESSOR = "send request to processor"
    START_PROCESSING = "start processing"
    INVALID_INPUT = "invalid input"
    PROCESS_SUCCESSFULLY_EXECUTED = "process successfully executed"
    PROCESS_STOPPED_WITH_ERRORS = "process finished with errors"
    PROCESS_SENT_TO_INGESTION = "process sent to ingestion service"
    PROCESS_INGESTED = "process ingested"


class PeriodicityEnum(Enum):
    DAILY = "daily"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class ProcessorType(Enum):
    SIMPLE = "simple"
    CROP = "crop"


def custom_asdict_factory(data):
    def convert_value(obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, date) or isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, list) and all(isinstance(objPath, Path) for objPath in obj):
            return [str(objPath) for objPath in obj]
        if isinstance(obj, list) and all(isinstance(objEnum, Enum) for objEnum in obj):
            return [objEnum.value for objEnum in obj]
        return obj

    return dict((k, convert_value(v)) for k, v in data)


@dataclass
class Message(object):
    message_id: str = field(default=None)
    process_id: str = field(default=None)
    processor: str = field(default=None)
    processor_type: ProcessorType = field(default=None)
    is_external_docker_processing: bool = field(default=False)
    internal_input_folder_path: Path = field(default=None)
    internal_output_folder_path: Path = field(default=None)
    external_folder_path: str = field(default=None)
    request_timestamp: float = field(default=None)
    status: ProcessingStatusEnum = field(default=None)
    user_email: str = field(default=None)
    processing_date: str = field(default=None)
    result_filename: str = field(default=None)
    periodicity: List[PeriodicityEnum] = field(default=None)
    processing_start_date: str = field(default=None)
    processing_end_date: str = field(default=None)
    result_configuration_path: Path = field(default=None)
    message: str = field(default=None)
    extended_perms: str = field(default=None)
    compact_perms: str = field(default=None)
    create_map_with_layers: bool = field(default=False)
    optional_fields: dict = field(default=None)
    map_creation_info: dict = field(default=None)
    is_process_success: bool = field(default=True)
    input_tiles: List[str] = field(default=None)
    outputs: List[Path] = field(default=None)
    output_styles: List[Path] = field(default=None)


    @staticmethod
    def from_json(json_dict):
        internal_input_folder_path = json_dict.get("internal_input_folder_path")
        internal_output_folder_path = json_dict.get("internal_output_folder_path")
        result_configuration_path = json_dict.get("result_configuration_path")
        periodicity = json_dict.get("periodicity")
        outputs = json_dict.get("outputs")
        output_styles = json_dict.get("output_styles")

        return Message(
            json_dict.get("message_id"),
            json_dict.get("process_id"),
            json_dict.get("processor"),
            json_dict.get("processor_type"),
            json_dict.get("is_external_docker_processing"),
            Path(internal_input_folder_path) if internal_input_folder_path is not None else None,
            Path(internal_output_folder_path) if internal_output_folder_path is not None else None,
            json_dict.get("external_folder_path"),
            json_dict.get("request_timestamp"),
            json_dict.get("status"),
            json_dict.get("user_email"),
            json_dict.get("processing_date"),
            json_dict.get("result_filename"),
            [PeriodicityEnum(value) for value in periodicity] if periodicity is not None else None,
            json_dict.get("processing_start_date"),
            json_dict.get("processing_end_date"),
            Path(result_configuration_path) if result_configuration_path is not None else None,
            json_dict.get("message"),
            json_dict.get("extended_perms"),
            json_dict.get("compact_perms"),
            json_dict.get("create_map_with_layers"),
            json_dict.get("optional_fields"),
            json_dict.get("map_creation_info"),
            json_dict.get("is_process_success"),
            json_dict.get("input_tiles"),
            [Path(value) for value in outputs] if outputs is not None else None,
            [Path(value) for value in output_styles] if output_styles is not None else None
        )






