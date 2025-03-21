from abc import abstractmethod
from typing import Any, Self
from cv2.typing import MatLike
from dataclasses import dataclass
from lib.image import convolve, expand, gaussianPyramid, laplacianPyramid, reconstruct, reduce
from lib.util import load_image, load_kernel, save_image


def filter_keys(d: dict, keys: list[str], deny_list=False):
    if deny_list:
        return {k: v for k, v in d.items() if k not in keys}
    else:
        return {k: v for k, v in d.items() if k in keys}


def build_operation(op: dict):
    if not "operation" in op:
        raise ValueError("Operation not specified")

    match op["operation"]:
        case "loadImage":
            return LoadImage(**filter_keys(op, ["name"]))
        case "saveImage":
            return SaveImage(**filter_keys(op, ["name"]))
        case "convolve":
            return Convolve(**filter_keys(op, ["kernel", "mode"]))
        case "reduce":
            return Reduce(**filter_keys(op, []))
        case "expand":
            return Expand(**filter_keys(op, []))
        case "gaussianPyramid":
            return GaussianPyramid(**filter_keys(op, ["levels"]))
        case "laplacianPyramid":
            return LaplacianPyramid(**filter_keys(op, ["levels"]))
        case "reconstruct":
            return Reconstruct(**filter_keys(op, ["levels"]))
        case _:
            raise ValueError(f"Unknown operation: {op['operation']}")


class Executable:
    @abstractmethod
    def pipe(self, image: MatLike) -> Self:
        return self

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass


@dataclass(kw_only=True)
class LoadImage(Executable):
    name: str

    def pipe(self, image: MatLike):
        # do nothing
        return super().pipe(image)

    def execute(self):
        return load_image(self.name)


@dataclass(kw_only=True)
class SaveImage(Executable):
    name: str
    image: MatLike = None

    def pipe(self, image: MatLike):
        self.image = image
        return self

    def execute(self):
        save_image(self.image, self.name)


@dataclass(kw_only=True)
class Convolve(Executable):
    kernel: str
    image: MatLike = None
    mode: str = "reflect"

    def __post_init__(self):
        self.kernel = load_kernel(self.kernel)

    def pipe(self, image: MatLike):
        self.image = image
        return self

    def execute(self):
        return convolve(self.image, self.kernel, self.mode)


@dataclass(kw_only=True)
class Reduce(Executable):
    image: MatLike = None

    def pipe(self, image: MatLike):
        self.image = image
        return self

    def execute(self):
        return reduce(self.image)


@dataclass(kw_only=True)
class Expand(Executable):
    image: MatLike = None

    def pipe(self, image: MatLike):
        self.image = image
        return self

    def execute(self):
        return expand(self.image)


@dataclass(kw_only=True)
class LaplacianPyramid(Executable):
    levels: int
    image: MatLike = None

    def pipe(self, image: MatLike):
        self.image = image
        return self

    def execute(self):
        return laplacianPyramid(self.image, self.levels)


@dataclass(kw_only=True)
class GaussianPyramid(Executable):
    levels: int
    image: MatLike = None

    def pipe(self, image: MatLike):
        self.image = image
        return self

    def execute(self):
        return gaussianPyramid(self.image, self.levels)


@dataclass(kw_only=True)
class Reconstruct(Executable):
    levels: int
    image: MatLike = None

    def pipe(self, image: MatLike):
        self.image = image
        return self

    def execute(self):
        return reconstruct(self.image, self.levels)
