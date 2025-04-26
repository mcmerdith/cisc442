import cv2 as cv
import numpy as np

from cv2.typing import MatLike

from lib.common import show_image
from lib.image import ScoreFunction, harris, normalize, overlay


def feature_based(left_image: MatLike, right_image: MatLike, window_size: tuple[int, int], search_area: int, score_fn: ScoreFunction):
    h, w = left_image.shape[:2]
    disparity_map = np.zeros((h, w), dtype=np.int32)

    half_h, half_w = window_size[0] // 2, window_size[1] // 2
    harris_left = harris(left_image)
    harris_right = harris(right_image)

    def get_patch(img, x, y):
        start_y = y - half_h
        end_y = y + half_h + 1
        start_x = x - half_w
        end_x = x + half_w + 1
        return img[start_y:end_y, start_x:end_x]

    def in_bounds(x, y):
        start_y = y - half_h
        end_y = y + half_h + 1
        start_x = x - half_w
        end_x = x + half_w + 1
        return start_y >= 0 and end_y < h and start_x >= 0 and end_x < w

    right_kp_dict = {(x, y): get_patch(right_image, x, y)
                     for x, y in harris_right
                     if in_bounds(x, y)}
    matched_image = np.hstack([img.copy()
                              for img in (overlay(left_image, harris_left), overlay(right_image, harris_right))])
    for (x_l, y_l) in harris_left:
        if not in_bounds(x_l, y_l):
            continue

        patch = get_patch(left_image, x_l, y_l)

        best_score = None
        best_match = None

        for (x_r, y_r), patch_right in right_kp_dict.items():
            if np.abs(x_l - x_r) > search_area or np.abs(y_l - y_r) > search_area:
                continue  # only match along same row

            assert patch_right.shape == patch.shape

            score = score_fn(patch_right, patch)
            if best_score is None or score > best_score:
                best_score = score
                best_match = (x_r, y_r)

        if best_match is not None:
            x_r, y_r = best_match
            disparity = np.abs(x_l - x_r)
            disparity_map[y_l, x_l] = disparity
            cv.line(matched_image, (x_l, y_l),
                    (x_r + w, y_r), (0, 255, 0), 1)

    show_image(matched_image)

    disparity_map = cv.inpaint(
        normalize(disparity_map), (disparity_map == 0).astype(np.uint8), 3, cv.INPAINT_TELEA)

    return disparity_map


# def interpolate(sparse: MatLike):
#     h, w = sparse.shape[:2]
#     ys, xs = np.nonzero(sparse)
#     values = sparse[ys, xs]

#     grid_y, grid_x = np.mgrid[0:h, 0:w]

#     dense_disp = griddata
