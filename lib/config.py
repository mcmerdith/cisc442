from dataclasses import dataclass
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


@dataclass
class TestOptions:
    save_images: bool


@dataclass
class Options:
    testing: TestOptions
    image_dir: str
    kernel_dir: str
    output_dir: str

    def __post_init__(self):
        self.testing = TestOptions(**self.testing)


@dataclass
class Config:
    options: Options
    execute: list[dict]

    def __post_init__(self):
        self.options = Options(**self.options)


def read_config(file: str):
    with open(file, "r") as f:
        try:
            return Config(**load(f, Loader=Loader))
        except TypeError as exc:
            exc.add_note("Invalid configuration option")
            raise RuntimeError(
                f"Failed to load config file: {file}"
            ) from exc
