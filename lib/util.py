from cv2.typing import MatLike
import cv2 as cv
import numpy as np
from lib.config import Config
from os import path, makedirs


def init(config: Config):
    global IMAGE_DIR, KERNEL_DIR, OUT_DIR
    IMAGE_DIR = config.options.image_dir
    KERNEL_DIR = config.options.kernel_dir
    OUT_DIR = config.options.output_dir


def save_image(image: MatLike, name: str):
    """
    Save an image to the output directory.

    Args:
        image (MatLike): The image to save
        name (str): The name of the image
    """
    makedirs(OUT_DIR, exist_ok=True)
    cv.imwrite(path.join(OUT_DIR, name), image)


def load_image(name: str):
    """
    Load an image from the image directory.

    Args:
        name (str): The name of the image

    Returns:
        MatLike: The loaded image
    """
    return cv.imread(path.join(IMAGE_DIR, name))


def load_kernel(name: str):
    return np.loadtxt(path.join(KERNEL_DIR, name))


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
