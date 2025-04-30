from time import time
import cv2 as cv
import numpy as np

from cv2.typing import MatLike

from lib.common import TaskTimer, console, get_image_sets, load_image_set, pair_images, prompt, save_image, show_image
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

    pairs = pair_images(load_image_set(image_set))

    process_pairs(method, image_set, pairs, template_x_size, template_y_size, search_range, score_fn)


def automatic():
    # image_sets = [(name, pair_images(load_image_set(name))) for name in get_image_sets()]
    image_sets = [("tsukuba", pair_images(load_image_set("tsukuba")))]

    methods = ["region", "feature"]
    method = "region"
    template_x_size, template_y_size = 5, 5
    search_range = 10
    score_fns = ["sad", "ssd", "ncc"]
    score_fn = "ncc"

    timer = TaskTimer(show_status=False)
    for name, pairs in image_sets:

        # for method in methods:
        #     for score_fn in score_fns:
                timer.start(f"Processing image set {name} ({method}-{score_fn})")
                process_pairs(method, name, pairs, template_x_size, template_y_size, search_range, score_fn)
                timer.complete()
        


def fill_gaps(disparity: MatLike, max_iterations: int = 20, max_size: int = 21, minimum_neighbors: int = 5):
    for _ in range(max_iterations):
        disparity = average_neighborhood(
            disparity, max_size, minimum_neighbors)
        if not np.any(disparity == 0):
            break
    return disparity


def process_pairs(method: str, name: str, image_pairs: list[tuple[MatLike, MatLike]], template_x_size: int, template_y_size: int, search_range: int, score_fn: str):
    timer = TaskTimer(show_status=False)
    for i, (left_image, right_image) in enumerate(image_pairs):
        timer.start(f"Processing pair {i+1}/{len(image_pairs)}")

        disparity = run(method, left_image, right_image, template_x_size,
                        template_y_size, search_range, score_fn)

        timer.complete()

        save_image(f"disparity_{i}.png", disparity, [name, method, score_fn])

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


automatic()
