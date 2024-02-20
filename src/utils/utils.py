import json


def print_json(data: dict | list | str) -> None:
    """
    Pretty prints the given dictionary or list.

    Args:
        data (dict | list): The data to pretty print.
    """
    if isinstance(data, str):
        data = json.loads(data)
    print(json.dumps(data, indent=4, sort_keys=True))