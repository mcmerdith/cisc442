import os
import cv2 as cv
import numpy as np

from cv2.typing import MatLike

from lib.common import load_image, prompt, save_image, score_NCC, score_SAD, score_SSD
from lib.region_matcher import region_based


method = "region"
left_image_path = "tsukuba/scene1.row3.col1.ppm"
right_image_path = "tsukuba/scene1.row3.col2.ppm"
left_image = load_image(left_image_path)
right_image = load_image(right_image_path)


def average_neighborhood(disparity: MatLike):
    h, w = disparity.shape[:2]

    averaged = disparity.copy()

    base_size = 3
    half_base_size = base_size // 2
    for y in range(half_base_size, h - half_base_size):
        for x in range(half_base_size, w - half_base_size):
            if np.abs(disparity[y, x]) > 0:
                continue
            # reset size
            size = base_size
            while True:
                half_size = size // 2

                # get neighborhood (clamped to edges)
                start_y = max(y - half_size, 0)
                end_y = min(y + half_size + 1, h)
                start_x = max(x - half_size, 0)
                end_x = min(x + half_size + 1, w)
                neighborhood = disparity[start_y:end_y,
                                         start_x:end_x]

                assert (neighborhood >= 0).all()
                non_zero = neighborhood[np.nonzero(neighborhood)]
                if non_zero.size < 5:
                    size += 2
                else:
                    break
            averaged[y, x] = np.mean(non_zero)

    return averaged


def feature_based(left_image, right_image, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y, disparity):
    pass


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


if method == 'region':
    disparity_ltr, disparity_rtl = region_based(
        left_image, right_image, (template_y_size, template_x_size), search_range, score_fn)
elif method == 'feature':
    disparity_ltr = feature_based()

# validation
diff = np.abs(disparity_ltr - disparity_rtl)
disparity_ltr[diff > 0] = 0
disparity_rtl[diff > 0] = 0

# fill gaps
for i in range(2):
    disparity_ltr = average_neighborhood(disparity_ltr)
    disparity_rtl = average_neighborhood(disparity_rtl)

diff = np.square(disparity_ltr - disparity_rtl)

disparity_ltr = cv.normalize(disparity_ltr, None, alpha=0,
                             beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
save_image('disparity.png', disparity_ltr)
