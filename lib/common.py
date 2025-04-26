from typing import Any, Callable, TypeVar
import cv2 as cv
import numpy as np

from cv2.typing import MatLike

from os import path, makedirs

ResponseType = TypeVar("ResponseType")


def prompt(message: str, options: list[ResponseType] = None, default: int | ResponseType = None, *, transformer: Callable[[str], ResponseType] = str, validator: Callable[[ResponseType], bool] = None) -> ResponseType:
    postfix = ""
    if options is not None:
        options = [i.lower() for i in options]
        postfix = f" ({", ".join(options)})"
        if default is not None:
            assert isinstance(
                default, int), "default must be an index when options are provided"
            assert default < len(
                options), f"index {default} out of range for provided options (length {len(options)})"
            default = options[default]
    if default is not None:
        postfix += f" [{str(default).upper()}]"

    while True:
        response = input(f"{message}{postfix}: ")

        if default is not None and response == "":
            response = default
            break

        if transformer is not None:
            try:
                response = transformer(response)
            except:
                continue
        elif options is not None:
            response = response.strip().lower()

        if validator is not None and not validator(response):
            continue

        if options is not None:
            if response in options:
                break
        else:
            break

    return response


def show_image(image: MatLike):
    cv.imshow("image", image)
    while True:
        key = cv.waitKey(100)
        # space or escape
        if key == 32 or key == 27:
            break
        if cv.getWindowProperty("image", cv.WND_PROP_VISIBLE) < 1:
            break
    cv.destroyWindow("image")


def load_image(filename: str):
    """
    Load an image as CV_U8
    """
    return cv.imread(path.join("input", filename))


def save_image(filename: str, image: MatLike):
    makedirs("output", exist_ok=True)
    cv.imwrite(path.join("output", filename), image)


def kp_overlay(image: MatLike, kp):
    return cv.drawKeypoints(image.copy(), kp, None)


def overlay(image: MatLike, points: np.ndarray):
    overlay = np.zeros(image.shape[:2], dtype=np.uint8)
    overlay[points[:, 1], points[:, 0]] = 1
    overlay = cv.dilate(overlay, None, iterations=2)

    overlayed = image.copy()
    overlayed[overlay == 1] = [0, 0, 255]

    return overlayed


def harris(image: MatLike):
    """
    Feature detection using Harris corner detector

    Returns a binary image of the corners
    """
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    corners = cv.cornerHarris(gray_image, 2, 3, 0.04)
    ret, corners = cv.threshold(corners, 0.01*corners.max(), 255, 0)
    corners = np.uint8(corners)

    ret, labels, stats, centroids = cv.connectedComponentsWithStats(corners)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    detections = np.int32(
        cv.cornerSubPix(gray_image, np.float32(
            centroids), (5, 5), (-1, -1), criteria)
    )

    return detections


ScoreFunction = Callable[[MatLike, MatLike], float]


def score_SAD(first: MatLike, second: MatLike):
    # this doesnt work for some reason
    return np.sum(np.abs(first - second))


def score_SSD(first: MatLike, second: MatLike):
    return np.sum(np.square(first - second))


def score_NCC(first: MatLike, second: MatLike):
    # questionable...
    m1 = np.mean(first)
    m2 = np.mean(second)

    numerator = np.sum((first - m1) * (second - m2))
    denominator = np.sqrt(np.sum(np.square(first - m1)) *
                          np.sum(np.square(second - m2)))

    if denominator == 0:
        return -1

    return -(numerator / denominator)
