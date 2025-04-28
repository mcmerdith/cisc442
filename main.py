import os
import cv2 as cv
import numpy as np

from cv2.typing import MatLike

from lib.common import load_image, prompt, save_image, show_image
from lib.feature_matcher import feature_based
from lib.image import ScoreFunction, average_neighborhood, normalize, score_NCC, score_SAD, score_SSD, validate
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


def run(method: str, left_image: MatLike, right_image: MatLike, template_x_size: int, template_y_size: int, search_range: int, score_fn: ScoreFunction):
    h, w = left_image.shape[:2]
    # left_image = cv.pyrDown(left_image)
    # right_image = cv.pyrDown(right_image)

    if method == 'region':
        disparity_ltr, disparity_rtl = region_based(
            left_image, right_image, (template_y_size, template_x_size), search_range, score_fn)

        # disparity_ltr = cv.resize(
        #     disparity_ltr, (w, h), interpolation=cv.INTER_NEAREST)
        # disparity_rtl = cv.resize(
        #     disparity_rtl, (w, h), interpolation=cv.INTER_NEAREST)

        show_image(normalize(disparity_ltr), name="disparity ltr")

        # validation
        disparity = validate(disparity_ltr, disparity_rtl)

        show_image(normalize(disparity), name="disparity")

        # fill gaps
        for _ in range(2):
            disparity = average_neighborhood(disparity, max_size=99)
    elif method == 'feature':
        disparity_ltr, disparity_rtl = feature_based(
            left_image, right_image, (template_x_size, template_y_size), search_range, score_fn)

        disparity = validate(disparity_ltr, disparity_rtl, threshold=1)
        show_image(np.hstack([normalize(d)
                   for d in (disparity_ltr, disparity, disparity_rtl)]), name="disparity raw")

        validated = disparity.copy()

        # for _ in range(2):
        #     disparity = average_neighborhood(disparity, max_size=99)

        averaged = disparity.copy()

        disparity = cv.inpaint(
            averaged, (averaged == 0).astype(np.uint8), 3, cv.INPAINT_TELEA)

        show_image(np.hstack([normalize(i)
                   for i in (validated, averaged, disparity)]), name="disparity")

    # disparity = cv.resize(
    #     cv.pyrUp(disparity), (w*2, h*2), interpolation=cv.INTER_NEAREST)

    disparity = normalize(disparity)

    show_image(disparity)
    save_image('disparity.png', disparity)


# left_image_path = "tsukuba/scene1.row3.col1.ppm"
# right_image_path = "tsukuba/scene1.row3.col2.ppm"
left_image_path = "barn1/im0.ppm"
right_image_path = "barn1/im1.ppm"
left_image = load_image(left_image_path)
right_image = load_image(right_image_path)

run("feature", left_image, right_image, 7, 7, 10, score_NCC)
