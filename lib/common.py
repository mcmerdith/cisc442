from os import makedirs, path
from time import time
from typing import Callable, TypeVar

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from lib.image import normalize


def load_image(filename: str):
    """
    Load an image as CV_U8
    """
    return cv.imread(path.join("input", filename))


def save_image(filename: str, image: MatLike):
    makedirs("output", exist_ok=True)
    cv.imwrite(path.join("output", filename), normalize(image))


def show_image(image: MatLike | list[MatLike], name: str = "image", timeout_sec: int = None):
    name = f"Preview: {name}"
    if isinstance(image, list):
        image = np.hstack([normalize(i) for i in image])
    else:
        image = normalize(image)
    cv.imshow(name, image)
    gui_wait_key(name, timeout_sec=timeout_sec)
    cv.destroyWindow(name)


def gui_wait_key(window_name: str, timeout_sec: int = None):
    opened = time()
    while True:
        key = cv.waitKey(100)
        if timeout_sec is not None:
            elapsed = time() - opened
            if elapsed > timeout_sec:
                break
            cv.setWindowTitle(
                window_name, f"{window_name} ({(timeout_sec - elapsed):.1f}s)")
        # space or escape
        if key == 32 or key == 27:
            break
        if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break


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
