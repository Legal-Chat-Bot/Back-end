from enum import Enum

class StatusCode(int, Enum):
    SUCCESS = 200
    UNAUTHORIZED = 401
    INVALID_REFRESH_TOKEN = 4401