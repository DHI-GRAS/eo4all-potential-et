from pathlib import Path


def get_resources_path() -> Path:
    return Path(__file__).parent / "resources"
