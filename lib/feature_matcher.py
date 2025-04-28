import cv2 as cv
import numpy as np

from cv2.typing import MatLike

from lib.common import show_image
from lib.image import ScoreFunction, harris, normalize, overlay


def feature_based(left_image: MatLike, right_image: MatLike, window_size: tuple[int, int], search_area: int, score_fn: ScoreFunction):
    h, w = left_image.shape[:2]
    disp_left = np.zeros((h, w), dtype=left_image.dtype)
    disp_right = np.zeros((h, w), dtype=right_image.dtype)

    half_h, half_w = window_size[0] // 2, window_size[1] // 2
    # harris_left = harris(left_image)
    # harris_right = harris(right_image)
    gray_left = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)

    harris_left = cv.cornerHarris(gray_left, 2, 3, 0.04)
    harris_right = cv.cornerHarris(gray_right, 2, 3, 0.04)

    # harris_left = cv.dilate(harris_left, None, iterations=1)
    # harris_right = cv.dilate(harris_right, None, iterations=1)

    harris_left[harris_left < 0.005*harris_left.max()] = 0
    harris_right[harris_right < 0.005*harris_right.max()] = 0
    harris_left[harris_left > 0] = 1
    harris_right[harris_right > 0] = 1

    show_image(np.hstack([normalize(i)
               for i in (harris_left, harris_right)]), name="harris")

    left_kp = np.argwhere(harris_left)
    right_kp = np.argwhere(harris_right)

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

    left_kp_dict = {(x, y): get_patch(gray_left, x, y)
                    for y, x in left_kp
                    if in_bounds(x, y)}
    right_kp_dict = {(x, y): get_patch(gray_right, x, y)
                     for y, x in right_kp
                     if in_bounds(x, y)}
    for (x_l, y_l), patch_left in left_kp_dict.items():
        # for (y_l, x_l) in left_kp:
        # y_l_start = y_l - half_h
        # y_l_end = y_l + half_h + 1
        # x_l_start = x_l - half_w
        # x_l_end = x_l + half_w + 1

        # if y_l_start < 0 or y_l_end > h or x_l_start < 0 or x_l_end > w:
        #     continue

        # patch_left = gray_left[y_l_start:y_l_end, x_l_start:x_l_end]

        best_score_left = None
        best_match_left = None
        best_score_right = None
        best_match_right = None

        for (x_r, y_r), patch_right in right_kp_dict.items():
            # for (y_r, x_r) in right_kp:
            # y_r_start = y_r - half_h
            # y_r_end = y_r + half_h + 1
            # x_r_start = x_r - half_w
            # x_r_end = x_r + half_w + 1

            # if y_r_start < 0 or y_r_end > h or x_r_start < 0 or x_r_end > w:
            #     continue
            if np.abs(x_l - x_r) > search_area or np.abs(y_l - y_r) > search_area:
                continue  # only match along same row

            # patch_right = gray_right[y_r_start:y_r_end, x_r_start:x_r_end]

            assert patch_right.shape == patch_left.shape

            score_left = score_fn(patch_left, patch_right)
            if best_score_left is None or score_left < best_score_left:
                best_score_left = score_left
                best_match_left = (x_r, y_r)

            score_right = score_fn(patch_right, patch_left)
            if best_score_right is None or score_right < best_score_right:
                best_score_right = score_right
                best_match_right = (x_r, y_r)

        if best_match_left is not None:
            x_r, y_r = best_match_left
            disparity = np.abs(x_l - x_r)
            disp_left[y_l, x_l] = disparity

        if best_match_right is not None:
            x_r, y_r = best_match_right
            disparity = np.abs(x_l - x_r)
            disp_right[y_l, x_l] = disparity

    disp_left = cv.dilate(normalize(disp_left), None)
    disp_right = cv.dilate(normalize(disp_right), None)

    disp_left = cv.GaussianBlur(disp_left, window_size, window_size[0]-1/6.0)
    disp_right = cv.GaussianBlur(disp_right, window_size, window_size[0]-1/6.0)

    return disp_left, disp_right


# def interpolate(sparse: MatLike):
#     h, w = sparse.shape[:2]
#     ys, xs = np.nonzero(sparse)
#     values = sparse[ys, xs]

#     grid_y, grid_x = np.mgrid[0:h, 0:w]

#     dense_disp = griddata
