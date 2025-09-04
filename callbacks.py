import os

from dotenv import load_dotenv, find_dotenv
from langfuse.langchain import CallbackHandler  # type: ignore

load_dotenv(find_dotenv())


def get_langfuse_handler():
    return CallbackHandler(
    )

