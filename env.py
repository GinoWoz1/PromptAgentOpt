import os
from typing import Any

from dotenv import load_dotenv

load_dotenv("./.env", override=True)


class MissingEnvError(Exception):
    """Exception raised when trying to grab environment variables that aren't set

    Attributes:
        envName -- the name of the environment variable being fetched
    """

    def __init__(self, env_name: str) -> None:
        message = f"missing required environment variable {env_name}"
        super().__init__(message)


def getenv(name: str, default: Any = None) -> str:
    """
    Gets an environment variable, or sets a default value.

        Parameters:
            name (str): The name of the environment variable to fetch
            default (str | None): The default value, if the env var isn't set. A value
                of None implies this env var is required.

        Returns:
            val (str): the env var

        Raises:
            MissingEnvError: when the env var wasn't set,
                and the default was not provided.
    """
    val = os.environ.get(name)
    if val is None:
        if default is None:
            raise MissingEnvError(name)
        return default
    return val
