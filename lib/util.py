import logging
from os import makedirs, path
from typing import Any, Iterable, Literal, Union, get_args, get_origin

import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from rich.console import Console
from rich.logging import RichHandler

from lib.config import Config

console = Console()
logging.basicConfig(format="%(message)s",
                    datefmt="[%X]", handlers=[RichHandler(console=console)])
logger = logging.getLogger("PR1")


def init(config: Config):
    global IMAGE_DIR, KERNEL_DIR, TEST_DIR, OUT_DIR, logger
    IMAGE_DIR = config.options.image_dir
    KERNEL_DIR = config.options.kernel_dir
    TEST_DIR = config.options.test_dir
    OUT_DIR = config.options.output_dir
    logger.setLevel(config.options.log_level.value)


def show_image(image: MatLike | list[MatLike]):
    """
    Show an image.

    Args:
        image (MatLike): The image to show
    """
    cv.imshow("Image", image)
    cv.waitKey(0)
    cv.destroyWindow("Image")


def save_image(image: list[MatLike] | MatLike, name: str):
    """
    Save an image to the output directory.

    Args:
        image (MatLike): The image to save
        name (str): The name of the image
    """
    makedirs(OUT_DIR, exist_ok=True)

    if isinstance(image, list):
        for i, img in enumerate(image):
            cv.imwrite(path.join(OUT_DIR, f"{name}_{i}.png"), img)
    else:
        cv.imwrite(path.join(OUT_DIR, name), image)


def load_image(name: str, test=False):
    """
    Load an image from the image directory.

    Args:
        name (str): The name of the image

    Returns:
        MatLike: The loaded image
    """
    if test:
        f = path.join(TEST_DIR, name)
    else:
        f = path.join(IMAGE_DIR, name)

    assert path.exists(f) and path.isfile(f), f"File not found: {f}"

    return cv.imread(f)


def load_kernel(name: str):
    """
    Load an kernel from the kernel directory.

    Args:
        name (str): The name of the kernel

    Returns:
        MatLike: The loaded kernel
    """
    return np.atleast_2d(np.loadtxt(path.join(KERNEL_DIR, name)))


def gaussian_kernel_1d(size: int, sigma=1.0) -> MatLike:
    """
    Create a 1D Gaussian kernel.

    Args:
        size (int): The size of the kernel
        sigma (float): The standard deviation of the kernel

    Returns:
        MatLike: The kernel
    """
    x = np.linspace(-(size // 2), (size // 2), size)
    gauss = 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-1*(x)**2/(2*sigma**2))

    return gauss


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
    """
    The worlds best sketchy runtime type checker.

    Probably catchs 99% of cases

    Args:
        value (Any): The value to check
        dtype (type): The type to check against

    Returns:
        bool: True if the value is of the type or not determinable, False otherwise
    """
    logger.debug(
        f"type checking: {type_name(type(value))} --IS-- {type_name(dtype)}")
    if dtype is Any:
        return True

    base_type = get_origin(dtype)
    args = get_args(dtype)

    if base_type is Literal:
        logger.debug(f"  Should be a literal of {args}")
        # literal type comparison
        return value in dtype.__args__
    elif base_type is Union:
        logger.debug(f"  Should be one of {args}")
        # recursively check all union types for the forsaken nested union
        return any([is_type(value, arg) for arg in args])
    elif base_type is dict:
        logger.debug(f"  Should be dict of {args}")
        if not isinstance(value, dict):
            return False
        if (len(args) == 0):
            # dont care about key or value type
            return True
        # recursively check all keys
        for k in value.keys():
            if not is_type(k, args[0]):
                return False
        if len(args) == 1:
            # dont care about value type
            return True

        # recursively check all values
        for v in value.values():
            if not is_type(v, args[1]):
                return False

        return True
    elif base_type is not None and issubclass(base_type, Iterable):
        logger.debug(f"  Should be iterable of {args}")
        if not isinstance(value, Iterable):
            # must actually be a list
            return False
        if len(args) == 0:
            # dont care about element type
            return True
        # recursively check all elements
        for v in value:
            if not is_type(v, args[0]):
                return False

        return True
    else:
        try:
            logger.debug(f"  Should be instance of {dtype}")
            # simple type comparison
            return isinstance(value, dtype)
        except:
            if base_type is not None:
                # fall back to really bad runtime type-checking
                logger.debug(f"    Alternative: {dtype}")
                isinstance(value, base_type)
            else:
                # no idea what this is
                logger.debug(f"  Failed to check type! {dtype}")
                return True
