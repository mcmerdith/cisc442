from dataclasses import dataclass
from enum import Enum
import logging
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class LogLevel(Enum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


@dataclass
class TestOptions:
    save_images: bool = False


@dataclass
class Options:
    testing: TestOptions
    image_dir: str = "images"
    kernel_dir: str = "kernels"
    test_dir: str = "test_input"
    output_dir: str = "output"
    log_level: LogLevel = LogLevel.WARNING

    def __post_init__(self):
        self.testing = TestOptions(**self.testing)
        self.log_level = LogLevel[self.log_level]


@dataclass
class Config:
    options: Options
    execute: list[dict]

    def __post_init__(self):
        self.options = Options(**self.options)


def read_config(file: str):
    """
    Read a config file and return a Config object.

    Args:
        file (str): The name of the config file

    Returns:
        Config: The loaded config
    """
    if not file.endswith(".yml") and not file.endswith(".yaml"):
        file = file + ".yml"
    with open(file, "r") as f:
        try:
            return Config(**load(f, Loader=Loader))
        except TypeError as exc:
            exc.add_note("Invalid configuration option")
            raise RuntimeError(
                f"Failed to load config file: {file}"
            ) from exc
