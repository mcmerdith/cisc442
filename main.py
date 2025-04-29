import cv2 as cv
import numpy as np

from cv2.typing import MatLike

from lib.common import load_image, prompt, save_image, show_image
from lib.feature_matcher import feature_based
from lib.image import average_neighborhood, normalize, validate
from lib.stereo import ScoreFunction, score_NCC, score_SAD, score_SSD
from lib.region_matcher import region_based


def interactive():
    method = prompt("Enter method", options=["region", "feature"], default=1)
    score_fn = prompt("Enter distance", options=[
                      "sad", "ssd", "ncc"], default=1)
    if score_fn == "sad":
        score_fn = score_SAD
    elif score_fn == "ssd":
        score_fn = score_SSD
    elif score_fn == "ncc":
        score_fn = score_NCC
    else:
        raise ValueError(f"Invalid score function: {score_fn}")
    search_range = prompt("Enter search range", transformer=int, default=16)
    template_x_size = prompt("Enter template_x_size (must be odd)", default=5,
                             transformer=int, validator=lambda x: x % 2 == 1)
    template_y_size = prompt("Enter template_y_size (must be odd)", default=5,
                             transformer=int, validator=lambda x: x % 2 == 1)


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
    h, w = left_image.shape[:2]
    # left_image = cv.pyrDown(left_image)
    # right_image = cv.pyrDown(right_image)
    # left_image = cv.resize(left_image, (w//2, h//2))
    # right_image = cv.resize(right_image, (w//2, h//2))

    if method == 'region':
        disparity_ltr, disparity_rtl = region_based(
            left_image, right_image, (template_y_size, template_x_size), search_range, score_fn)

        # validation
        disparity = validate(disparity_ltr, disparity_rtl)

        # fill gaps
        while np.any(disparity == 0):
            disparity = average_neighborhood(disparity)
    elif method == 'feature':
        disparity_ltr, disparity_rtl = feature_based(
            left_image, right_image, (template_x_size, template_y_size), search_range, score_fn)

        # validation
        disparity = validate(disparity_ltr, disparity_rtl)

        original = disparity.copy()

        # fill gaps
        disparity = fill_gaps(disparity)

        show_image([original, disparity])

        # # fill the gaps
        # disparity = normalize(disparity)
        # disparity = cv.inpaint(
        #     disparity, (disparity == 0).astype(np.uint8), 3, cv.INPAINT_TELEA)

    disparity = normalize(disparity)

    save_image(f'disparity_{method}_{score_fn}.png', disparity)


# left_image_path = "tsukuba/scene1.row3.col1.ppm"
# right_image_path = "tsukuba/scene1.row3.col2.ppm"
left_image_path = "barn1/im0.ppm"
right_image_path = "barn1/im1.ppm"
left_image = load_image(left_image_path)
right_image = load_image(right_image_path)

run("feature", left_image, right_image, 7, 7, 10, "ncc")
