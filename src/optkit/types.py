from enum import Enum

class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"

def normalize_option_type(option_type: str) -> str:
    ot = option_type.lower().strip()
    if ot not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    return ot
