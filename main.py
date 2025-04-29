from time import time
import cv2 as cv
import numpy as np

from cv2.typing import MatLike

from lib.common import TaskTimer, console, get_image_sets, load_image_set, prompt, save_image, show_image
from lib.feature_matcher import feature_based
from lib.image import average_neighborhood, normalize, validate
from lib.region_matcher import region_based


def interactive():
    image_set = prompt("Select image set", options=get_image_sets(), default=0)
    method = prompt("Select method", options=["region", "feature"], default=0)
    score_fn = prompt("Select score function", options=[
                      "sad", "ssd", "ncc"], default=1)
    search_range = prompt("Enter search range", transformer=int, default=10)
    template_x_size = prompt("Enter template_x_size (must be odd)", default=7,
                             transformer=int, validator=lambda x: x % 2 == 1)
    template_y_size = prompt("Enter template_y_size (must be odd)", default=7,
                             transformer=int, validator=lambda x: x % 2 == 1)

    image_set = load_image_set(image_set)

    pairs = [(image_set[i], image_set[i+1])
             for i in range(len(image_set)-1)]

    timer = TaskTimer(show_status=False)
    for i, (left_image, right_image) in enumerate(pairs):
        timer.start(f"Processing pair {i+1}/{len(pairs)}")

        disparity = run(method, left_image, right_image, template_x_size,
                        template_y_size, search_range, score_fn)

        timer.complete()

        save_image(f'disparity_{method}_{score_fn}_{i}.png', disparity)


def automatic():
    pass


def fill_gaps(disparity: MatLike, max_iterations: int = 20, max_size: int = 21, minimum_neighbors: int = 5):
    for _ in range(max_iterations):
        disparity = average_neighborhood(
            disparity, max_size, minimum_neighbors)
        if not np.any(disparity == 0):
            break
    return disparity


def run(method: str, left_image: MatLike, right_image: MatLike, template_x_size: int, template_y_size: int, search_range: int, score_fn: str):
    timer = TaskTimer().start("Calculating disparity")

    if method == 'region':
        disparity_ltr, disparity_rtl = region_based(
            left_image, right_image, (template_y_size, template_x_size), search_range, score_fn)
    elif method == 'feature':
        disparity_ltr, disparity_rtl = feature_based(
            left_image, right_image, (template_x_size, template_y_size), search_range, score_fn)

    timer.complete().start("Validating disparity")

    # validation
    disparity = validate(disparity_ltr, disparity_rtl)

    timer.complete().start("Filling gaps")

    # fill gaps
    disparity = fill_gaps(disparity)

    timer.complete()

    return normalize(disparity)


interactive()
