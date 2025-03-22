from abc import abstractmethod
import dataclasses
from typing import Any, Literal, Self, TypeVar, Union, get_args, get_origin
from cv2.typing import MatLike
from dataclasses import dataclass, field

import numpy as np
from lib.config import LogLevel
from lib.image import convolve, expand, gaussianPyramid, laplacianPyramid, reconstruct, reduce
from lib.util import load_image, load_kernel, log, save_image

MatLikeArgs = set(get_args(MatLike))


def type_name(dtype: type):
    """
    Get the most readable name for a type.

    Mostly to make type hints for MatLike arguments more readable.

    Args:
        dtype (type): The type to get the name for.

    Returns:
        str: The name of the type.
    """
    base_type = get_origin(dtype)
    if base_type is Union:
        args = get_args(dtype)
        argset = set(args)

        # Strip out "MatLike"
        has_image = argset & MatLikeArgs == MatLikeArgs
        if has_image:
            argset = argset - MatLikeArgs

        type_str = ["None" if arg is None else arg.__name__
                    for arg in args
                    if arg in argset]

        if has_image:
            type_str.insert(0, "Pipe[Image]")

        return " | ".join(type_str)
    elif dtype is None:
        return "None"
    else:
        return dtype.__name__


def is_type(value: Any, dtype: type):
    base_type = get_origin(dtype)
    valid = False
    if base_type is Literal:
        # literal type comparison
        valid = value in dtype.__args__
    elif base_type is None or base_type is Union:
        # simple type comparison
        valid = isinstance(value, dtype)
    else:
        # fall back to really bad runtime type-checking
        valid = isinstance(value, base_type)

    return valid


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
            if not isinstance(data, required_type):
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
        return type(self).__name__ + "[" + self._id + "]"

    def print(self, *args, level=LogLevel.INFO, **kwargs):
        log(self.get_name(), *args, level=level, **kwargs)

    def status(self, *args, **kwargs):
        self.print(*args, level=LogLevel.DEBUG, **kwargs)


def filter_keys(d: dict, keys: list[str], deny_list=False):
    """
    Filter a dict to include or exclude certain keys

    Args:
        d (dict): The dict to filter
        keys (list[str]): The keys to include or exclude
        deny_list (bool): If True, exclude the keys, otherwise include them

    Returns:
        dict: The filtered dict
    """
    if deny_list:
        return {k: v for k, v in d.items() if k not in keys}
    else:
        return {k: v for k, v in d.items() if k in keys}


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
        case "loadImage":
            return LoadImage(**filter_keys(op, ["name"]), _id=id)
        case "saveImage":
            return SaveImage(**filter_keys(op, ["name"]), _id=id)
        case "convolve":
            return Convolve(**filter_keys(op, ["kernel", "mode"]), _id=id)
        case "reduce":
            return Reduce(**filter_keys(op, []), _id=id)
        case "expand":
            return Expand(**filter_keys(op, []), _id=id)
        case "gaussianPyramid":
            return GaussianPyramid(**filter_keys(op, ["levels"]), _id=id)
        case "laplacianPyramid":
            return LaplacianPyramid(**filter_keys(op, ["levels"]), _id=id)
        case "reconstruct":
            return Reconstruct(**filter_keys(op, ["levels"]), _id=id)
        case "compare":
            return Compare(**filter_keys(op, ["reference", "test"]), _id=id)
        case _:
            raise ValueError(f"Unknown operation: {op['operation']}")


@dataclass(kw_only=True)
class LoadImage(Executable):
    name: str = None

    def execute(self):
        super().execute()
        self.status(f"Loading image: {self.name}")
        return load_image(self.name)


@dataclass(kw_only=True)
class SaveImage(Executable):
    name: str = None
    data: MatLike = None

    def execute(self):
        super().execute()
        self.status(f"Saving image: {self.name}")
        save_image(self.data, self.name)
        return self.data


@dataclass(kw_only=True)
class Convolve(Executable):
    kernel: str | MatLike = None
    data: MatLike = None
    mode: str = "reflect"

    def execute(self):
        super().execute()
        self.status(f"Convolving with kernel: {self.kernel}")
        if isinstance(self.kernel, str):
            self.kernel = load_kernel(self.kernel)
        return convolve(self.data, self.kernel, self.mode)


@dataclass(kw_only=True)
class Reduce(Executable):
    data: MatLike = None

    def execute(self):
        super().execute()
        self.status(f"Reducing image")
        return reduce(self.data)


@dataclass(kw_only=True)
class Expand(Executable):
    data: MatLike = None

    def execute(self):
        super().execute()
        self.status(f"Expanding image")
        return expand(self.data)


@dataclass(kw_only=True)
class LaplacianPyramid(Executable):
    levels: int = None
    data: MatLike = None

    def execute(self):
        super().execute()
        self.status(f"Computing Laplacian Pyramid with {self.levels} levels")
        return laplacianPyramid(self.data, self.levels)


@dataclass(kw_only=True)
class GaussianPyramid(Executable):
    levels: int = None
    data: MatLike = None

    def execute(self):
        super().execute()
        self.status(f"Computing Gaussian Pyramid with {self.levels} levels")
        return gaussianPyramid(self.data, self.levels)


@dataclass(kw_only=True)
class Reconstruct(Executable):
    levels: int = None
    data: MatLike = None

    def execute(self):
        super().execute()
        self.status(f"Reconstructing image with {self.levels} levels")
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

        self.print("Difference is", np.sum(np.abs(self.reference - self.data)))

        return np.allclose(self.reference, self.data)
