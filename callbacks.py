import os

from dotenv import load_dotenv, find_dotenv
# Added a check for langfuse import to prevent errors if the package is missing
try:
    from langfuse.langchain import CallbackHandler  # type: ignore
    LANGFUSE_AVAILABLE = True
except ImportError:
    CallbackHandler = None
    LANGFUSE_AVAILABLE = False
    # Use print as logger might not be configured yet.
    # We check if the root logger has handlers to avoid printing if logging is already set up.
    import logging
    if not logging.getLogger().hasHandlers():
        print("INFO: Langfuse SDK not installed. Tracing will be unavailable.")

load_dotenv(find_dotenv())


def get_langfuse_handler():
    if not LANGFUSE_AVAILABLE:
        # Raise an error if the handler is requested but the SDK is missing
        raise ImportError("Langfuse SDK not installed. Cannot initialize handler.")
    # The CallbackHandler automatically picks up keys from the environment variables
    # (LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST)
    return CallbackHandler(
    )