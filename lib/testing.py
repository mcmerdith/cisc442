from time import time
import cv2 as cv
import numpy as np
from lib.config import Config
from lib.pipeline import build_executor
from lib.util import gaussian_kernel_1d, load_image, save_image, logger, console
from lib.image import convolve, expand_image, laplacian_pyramid, reconstruct, reduce_image


SAVE_IMAGES = False


def test(config: Config):
    global SAVE_IMAGES
    SAVE_IMAGES = config.options.testing.save_images

    basic_tests = [test_utils, test_convolve,
                   test_reduce, test_expand, test_reconstruction]

    with console.status("Executing") as status:
        for test in basic_tests:
            t = time()
            test()
            logger.info(f"{test.__name__} complete in {time() - t} seconds")

    test_pipeline(config)


def test_utils():
    assert np.allclose(gaussian_kernel_1d(5),
                       [0.05399097, 0.24197072, 0.39894228, 0.24197072, 0.05399097])


def is_good_enough(a, b):
    """The magic function because I don't care about being exactly the same as opencv"""
    return np.allclose(a, b, atol=1e-2)


def test_convolve():
    image = load_image("lena.png")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # test kernel
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]).astype(np.float32)

    # execute operation
    convolved = convolve(image, sobel_x)
    gray_convolved = convolve(gray, sobel_x)

    if SAVE_IMAGES:
        save_image(convolved, "test_convolved.png")
        save_image(gray_convolved, "test_gray_convolved.png")

    # shape shouldn't change
    assert convolved.shape == image.shape
    assert gray_convolved.shape == gray.shape

    # test against open-cv
    cv_convolved = cv.filter2D(image, -1, sobel_x)
    cv_gray_convolved = cv.filter2D(gray, -1, sobel_x)
    assert is_good_enough(convolved, cv_convolved)
    assert is_good_enough(gray_convolved, cv_gray_convolved)

    logger.critical("Convolution OK")


def test_reduce():
    image = load_image("lena.png")

    reduced = reduce_image(image)
    if SAVE_IMAGES:
        save_image(reduced, "test_reduced.png")

    # image should be half the size
    assert reduced.shape[:2] == (image.shape[0] // 2, image.shape[1] // 2)

    logger.critical("Reduce OK")


def test_expand():
    image = load_image("lena.png")

    expanded = expand_image(image)
    if SAVE_IMAGES:
        save_image(expanded, "test_expanded.png")

    # image should be double the size
    assert expanded.shape[:2] == (image.shape[0] * 2, image.shape[1] * 2)

    logger.critical("Expand OK")


def test_reconstruction():
    image = load_image("lena.png")
    pyramid = laplacian_pyramid(image, 5)
    reconstructed = reconstruct(pyramid, 5)

    logger.critical(
        f"Reconstruction difference: {np.sum(image - reconstructed)}")


def test_pipeline(config: Config):
    logger.critical("Beginning pipeline test")

    executor = build_executor(config.execute)

    with console.status("Testing Pipeline") as status:
        assert all([step.execute() for step in executor]), "Pipeline Failure"

    logger.critical("Pipeline OK")
