from functools import lru_cache
from .DBConfig import DBConfig
from .LLMConfig import LLMConfig
from .APIConfig import APIConfig
from .Test import Test


class Config:
    def __init__(self):
        self.gdb = DBConfig()
        self.llm = LLMConfig()
        self.api = APIConfig()
        self.test = Test()


@lru_cache()
def get_config():
    return Config()