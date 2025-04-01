import cv2 as cv
import numpy as np
from lib.config import Config
from lib.pipeline import build_executor
from lib.util import gaussian_kernel_1d, load_image, save_image, logger, console
from lib.image import convolve, expand_image, reduce_image
from rich.progress import track


SAVE_IMAGES = False


def test(config: Config):
    global SAVE_IMAGES
    SAVE_IMAGES = config.options.testing.save_images

    basic_tests = [test_utils, test_convolve, test_reduce, test_expand]

    with console.status("Executing") as status:
        for test in basic_tests:
            test()

    test_pipeline(config)


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

    if SAVE_IMAGES:
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

    logger.info("Convolution OK")


def test_reduce():
    image = load_image("lena.png")

    reduced = reduce_image(image)
    if SAVE_IMAGES:
        save_image(reduced, "test_reduced.png")

    # image should be half the size
    assert reduced.shape[:2] == (image.shape[0] // 2, image.shape[1] // 2)

    logger.info("Reduce OK")


def test_expand():
    image = load_image("lena.png")

    expanded = expand_image(image)
    if SAVE_IMAGES:
        save_image(expanded, "test_expanded.png")

    # image should be double the size
    assert expanded.shape[:2] == (image.shape[0] * 2, image.shape[1] * 2)

    logger.info("Expand OK")


def test_pipeline(config: Config):
    logger.info("Beginning pipeline test")

    executor = build_executor(config.execute)

    with console.status("Testing Pipeline") as status:
        assert all([step.execute() for step in executor]), "Pipeline Failure"

    logger.info("Pipeline OK")
