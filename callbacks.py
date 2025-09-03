import os

from dotenv import load_dotenv
from langfuse.callback import CallbackHandler  # type: ignore

load_dotenv()


def get_langfuse_handler():
    return CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST"),
    )
