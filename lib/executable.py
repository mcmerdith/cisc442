import dataclasses
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Self, TypeVar

import numpy as np
from cv2.typing import MatLike

from lib.config import LogLevel
from lib.gui import PointMatcherGui, ShowImageGui
from lib.image import (convolve, expand_image, gaussian_pyramid,
                       laplacian_pyramid, mosaic_images, reconstruct,
                       reduce_image)
from lib.util import (is_type, load_image, load_kernel, logger, normalize_u8, save_image,
                      type_name, filter_keys)


@dataclass(kw_only=True)
class Executable:
    """
    An executable operation.

    Implementors should be a kw_only dataclass and implement the `execute` method.

    Data Validation:
        - Type hints are used to validate the data
        - All fields have a default value (reasonable or None)
        - Optional fields should be annotated as `Type | None`
    """
    data: Any = None
    _id: str = "?"
    _fields: dict[str, type] = field(
        init=False, compare=False, hash=False, repr=False)

    def __post_init__(self):
        fields = dataclasses.fields(self)
        self._fields = {
            field.name: field.type for field in fields
        }

    @abstractmethod
    def execute(self) -> Any:
        """
        Execute the operation.

        Returns:
            Any: The result of the operation.

        Notes: Must be overloaded and call super().execute() for automatic validation.
        """
        for field, dtype in self._fields.items():
            if field.startswith("_") or dtype is None or dtype is Any:
                continue

            display_name = self.get_name() + "." + field

            if not hasattr(self, field):
                raise ValueError(
                    f"Missing required argument `{display_name}`"
                )

            value = getattr(self, field)

            if not is_type(value, dtype):
                raise ValueError(
                    f"Invalid type for {display_name}: expected `{type_name(dtype)}`, got `{type_name(type(value))}`"
                )

    def pipe_in(self, data) -> Self:
        """
        Provide the input to the operation.

        Returns:
            Self: for convenience.

        Notes: Automatic type-guarding of the `data` class variable is performed.
        """

        required_type = self._fields.get("data", None)
        if required_type is not None and required_type is not Any:
            if not is_type(data, required_type):
                raise ValueError(
                    f"Invalid input type for {self.get_name()}: expected `{type_name(required_type)}`, got `{type_name(type(data))}` (broken pipe?)"
                )
        self.data = data
        return self

    def get_name(self):
        """
        The readable name and id of the operation.

        Returns:
            str: The name of the operation.
        """
        return self._id + "-" + type(self).__name__

    def info(self, *args, level=LogLevel.INFO, **kwargs):
        logger.log(
            level.value, f"[{self.get_name()}] {' '.join([str(arg) for arg in args])}", **kwargs
        )

    def debug(self, *args, **kwargs):
        self.info(*args, level=LogLevel.DEBUG, **kwargs)


ArgType = TypeVar("ArgType")


def build_operation(op: dict, id: str):
    """
    Create an operation from a config file entry

    Args:
        op (dict): The config file entry
        id (str): The id of the operation

    Returns:
        Executable: The operation
    """
    if not "operation" in op:
        raise ValueError("Operation not specified")

    match op["operation"]:
        case "load_image":
            return LoadImage(**filter_keys(op, ["name"]), _id=id)
        case "save_image":
            return SaveImage(**filter_keys(op, ["name"]), _id=id)
        case "show_image":
            return ShowImage(**filter_keys(op, ["timeout"]), _id=id)
        case "convolve":
            return Convolve(**filter_keys(op, ["kernel", "mode"]), _id=id)
        case "reduce":
            return Reduce(**filter_keys(op, []), _id=id)
        case "expand":
            return Expand(**filter_keys(op, []), _id=id)
        case "gaussian_pyramid":
            return GaussianPyramid(**filter_keys(op, ["levels"]), _id=id)
        case "laplacian_pyramid":
            return LaplacianPyramid(**filter_keys(op, ["levels"]), _id=id)
        case "reconstruct":
            return Reconstruct(**filter_keys(op, ["levels"]), _id=id)
        case "compare":
            return Compare(**filter_keys(op, ["reference", "test"]), _id=id)
        case "mosaic":
            return Mosaic(**filter_keys(op, ["source2", "points", "points2", "interactive"]), _id=id)
        case _:
            raise ValueError(f"Unknown operation: {op['operation']}")


#################################################
#             Executable Operations             #
# Mostly just wrappers around the image library #
#################################################


@dataclass(kw_only=True)
class LoadImage(Executable):
    name: str = None

    def execute(self):
        super().execute()
        self.info(f"Loading image: {self.name}")
        return load_image(self.name)


@dataclass(kw_only=True)
class SaveImage(Executable):
    name: str = None
    data: list[MatLike] | MatLike = None

    def execute(self):
        super().execute()
        self.info(f"Saving image: {self.name}")
        save_image(self.data, self.name)
        return self.data


@dataclass(kw_only=True)
class ShowImage(Executable):
    data: MatLike = None
    timeout: int | None = None

    def execute(self):
        super().execute()
        self.info(f"Showing image")
        ShowImageGui(image=self.data, timeout=self.timeout).init()
        return self.data


@dataclass(kw_only=True)
class Convolve(Executable):
    kernel: str | MatLike = None
    data: MatLike = None
    mode: str = "reflect"

    def execute(self):
        super().execute()
        self.info(f"Convolving with kernel: {self.kernel}")
        if isinstance(self.kernel, str):
            self.kernel = load_kernel(self.kernel)
        return convolve(self.data, self.kernel, self.mode)


@dataclass(kw_only=True)
class Reduce(Executable):
    data: MatLike = None

    def execute(self):
        super().execute()
        self.info(f"Reducing image")
        return reduce_image(self.data)


@dataclass(kw_only=True)
class Expand(Executable):
    data: MatLike = None

    def execute(self):
        super().execute()
        self.info(f"Expanding image")
        return expand_image(self.data)


@dataclass(kw_only=True)
class LaplacianPyramid(Executable):
    levels: int = None
    data: MatLike = None

    def execute(self):
        super().execute()
        self.info(f"Computing Laplacian Pyramid with {self.levels} levels")
        return laplacian_pyramid(self.data, self.levels)


@dataclass(kw_only=True)
class GaussianPyramid(Executable):
    levels: int = None
    data: MatLike = None

    def execute(self):
        super().execute()
        self.info(f"Computing Gaussian Pyramid with {self.levels} levels")
        return gaussian_pyramid(self.data, self.levels)


@dataclass(kw_only=True)
class Reconstruct(Executable):
    levels: int = None
    data: list[MatLike] = None

    def execute(self):
        super().execute()
        self.info(f"Reconstructing image with {self.levels} levels")
        return reconstruct(self.data, self.levels)


@dataclass(kw_only=True)
class Compare(Executable):
    reference: str | MatLike = None
    test: bool = False
    data: MatLike = None

    def execute(self):
        super().execute()
        if isinstance(self.reference, str):
            self.reference = load_image(self.reference, test=self.test)

        ref = normalize_u8(self.reference)
        data = normalize_u8(self.data)

        diff = np.abs(ref - data)

        self.info("Difference is", np.sum(diff))

        return np.allclose(ref, data)


@dataclass(kw_only=True)
class Mosaic(Executable):
    data: MatLike = None
    source2: MatLike | str = None
    points: list[list[int]] | None = None
    points2: list[list[int]] | None = None
    interactive: bool = False

    def execute(self):
        super().execute()
        self.info(f"Mosaicing images")
        if isinstance(self.source2, str):
            self.source2 = load_image(self.source2)

        if self.points and self.points2:
            p1, p2 = np.array(self.points), np.array(self.points2)
        elif self.interactive:
            matcher = PointMatcherGui(images=[self.data, self.source2]).init()
            p1, p2 = np.array(matcher.points[0]), np.array(matcher.points[1])
        else:
            p1, p2 = None, None

        mosaic = mosaic_images(self.data, self.source2, p1, p2)

        return mosaic
