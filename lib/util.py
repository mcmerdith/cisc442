from enum import Enum
from cv2.typing import MatLike
import cv2 as cv
import numpy as np
from lib.config import Config, LogLevel
from os import path, makedirs


def init(config: Config):
    global IMAGE_DIR, KERNEL_DIR, TEST_DIR, OUT_DIR, LOG_LEVEL
    IMAGE_DIR = config.options.image_dir
    KERNEL_DIR = config.options.kernel_dir
    TEST_DIR = config.options.test_dir
    OUT_DIR = config.options.output_dir
    LOG_LEVEL = config.options.log_level


def log(*args, level=LogLevel.INFO, **kwargs):
    if LOG_LEVEL.value <= level.value:
        print(*args, **kwargs)


def save_image(image: MatLike, name: str):
    """
    Save an image to the output directory.

    Args:
        image (MatLike): The image to save
        name (str): The name of the image
    """
    makedirs(OUT_DIR, exist_ok=True)

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
