import cv2 as cv
import numpy as np
from lib.util import gaussian_kernel_1d, load_image, save_image
from lib.image import convolve, expand, reduce

SAVE_TEST_IMAGES = True


def test():
    test_utils()
    test_convolve()
    test_reduce()
    test_expand()


def test_utils():
    assert np.allclose(gaussian_kernel_1d(5),
                       [0.05399097, 0.24197072, 0.39894228, 0.24197072, 0.05399097])


def test_convolve():
    image = load_image("lena.png")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # test kernel
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    # execute operation
    convolved = convolve(image, sobel_x)
    gray_convolved = convolve(gray, sobel_x)

    if SAVE_TEST_IMAGES:
        save_image(convolved, "test_convolved.png")
        save_image(gray_convolved, "test_gray_convolved.png")

    # shape shouldn't change
    assert convolved.shape == image.shape
    assert gray_convolved.shape == gray.shape

    # test against open-cv
    cv_convolved = cv.filter2D(image, -1, sobel_x)
    cv_gray_convolved = cv.filter2D(gray, -1, sobel_x)
    assert (convolved == cv_convolved).all()
    assert (gray_convolved == cv_gray_convolved).all()

    print("Convolution OK")


def test_reduce():
    image = load_image("lena.png")

    reduced = reduce(image)
    if SAVE_TEST_IMAGES:
        save_image(reduced, "test_reduced.png")

    # image should be half the size
    assert reduced.shape[:2] == (image.shape[0] // 2, image.shape[1] // 2)

    print("Reduce OK")


def test_expand():
    image = load_image("lena.png")

    expanded = expand(image)
    if SAVE_TEST_IMAGES:
        save_image(expanded, "test_expanded.png")

    # image should be double the size
    assert expanded.shape[:2] == (image.shape[0] * 2, image.shape[1] * 2)

    print("Expand OK")
