import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from lib.util import gaussian_kernel_1d


def convolve(I: MatLike, H: MatLike, mode='reflect') -> MatLike:
    """
    Perform convolution of an image with a given kernel.

    Args:
        I (MatLike): The input image in the shape (height, width, channels)
        H (MatLike): The kernel in the shape (height, width)
        mode (str): Mode for np.pad()

    Returns:
        MatLike: The result of the convolution
    """

    # accomodate grayscale and color images
    is_color = I.ndim == 3

    # get dimensions
    img_h, img_w = I.shape[:2]
    kernel_h, kernel_w = H.shape

    # compute required padding
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    # pad the image
    if is_color:
        # no padding for color channels
        padding = ((pad_h, pad_h), (pad_w, pad_w), (0, 0))
    else:
        padding = ((pad_h, pad_h), (pad_w, pad_w))

    # pad the image
    I_padded = np.pad(I, padding, mode=mode)

    # create a blank output image
    output = np.zeros(I.shape, dtype=np.float32)

    # do the convolution for each channel of each pixel
    for y in range(img_h):
        for x in range(img_w):
            if is_color:
                # apply to all channels for RGB
                for channel in range(I.shape[2]):
                    output[y, x, channel] = np.sum(
                        H * I_padded[y:y + kernel_h, x:x + kernel_w, channel]
                    )
            else:
                # apply to grayscale
                output[y, x] = np.sum(
                    H * I_padded[y:y + kernel_h, x:x + kernel_w]
                )

    # clip to valid range [0, 255] and cast to uint8
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output


def reduce(I: MatLike):
    """
    Uniformly reduce the scale of an image by 1/2.

    Gaussian Smoothing -> Resizing (area interpolation)

    Args:
        I (MatLike): The input image in the shape (height, width, [channels])

    Returns:
        MatLike: The reduced image
    """

    gaussian_1d = gaussian_kernel_1d(5)

    vertical_gaussian = gaussian_1d.reshape(-1, 1)
    smoothed = convolve(I, vertical_gaussian)

    horizontal_gaussian = gaussian_1d.reshape(1, -1)
    smoothed = convolve(smoothed, horizontal_gaussian)

    reduced = cv.resize(
        smoothed, (I.shape[1] // 2, I.shape[0] // 2), interpolation=cv.INTER_AREA
    )

    return reduced


def expand(I: MatLike):
    """
    Uniformly expand the scale of an image by 2 using cubic interpolation.

    Args:
        I (MatLike): The input image in the shape (height, width, [channels])

    Returns:
        MatLike: The expanded image
    """

    return cv.resize(I, (I.shape[1] * 2, I.shape[0] * 2), interpolation=cv.INTER_CUBIC)


def gaussianPyramid(I: MatLike, n: int):
    """
    Compute the n-level Gaussian pyramid of an image.

    Args:
        I (MatLike): The input image
        n (int): The number of levels

    Returns:
        list[MatLike]: The n-level Gaussian pyramid
    """

    levels = [I]
    for _ in range(n-1):
        levels.append(reduce(levels[-1]))

    return levels


def laplacianPyramid(I: MatLike, n):
    """
    Compute the n-level Laplacian pyramid of an image.

    Args:
        I (MatLike): The input image
        n (int): The number of levels

    Returns:
        list[MatLike]: The n-level Laplacian pyramid
    """

    pyramid = gaussianPyramid(I, n)

    levels = []
    for i in range(n-1):
        levels.append(pyramid[i] - expand(pyramid[i+1]))
    levels.append(pyramid[-1])

    return levels


def reconstruct(LI: list[MatLike], n):
    """
    Reconstruct an image from a Laplacian pyramid.

    Args:
        LI (list[MatLike]): The Laplacian pyramid
        n (int): The number of levels

    Returns:
        MatLike: The reconstructed image
    """
    assert len(LI) == n

    current_level = LI[-1]
    for i in reversed(range(n-1)):
        current_level = LI[i] + expand(current_level)

    return current_level
