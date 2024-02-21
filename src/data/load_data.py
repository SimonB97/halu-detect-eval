import json


def read_json(file_path: str) -> dict | list:
    """Reads a JSON file and returns its contents as a dictionary or list."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data