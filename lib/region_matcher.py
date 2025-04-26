import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from lib.image import ScoreFunction


def region_based(left_image: MatLike, right_image: MatLike, window_size: tuple[int, int], search_area: int, score_fn: ScoreFunction):
    gray_left = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)
    h, w = left_image.shape[:2]
    half_h, half_w = window_size[0] // 2, window_size[1] // 2

    disparity_ltr = np.zeros((h, w), dtype=np.int32)
    disparity_rtl = np.zeros((h, w), dtype=np.int32)

    for y in range(half_h, h - half_h):
        for x in range(half_w, w - half_w):
            start_y = y - half_h
            end_y = y + half_h + 1
            start_x = x - half_w
            end_x = x + half_w + 1

            t_left = gray_left[start_y:end_y, start_x:end_x]
            t_right = gray_right[start_y:end_y, start_x:end_x]

            best_score_ltr = None
            best_score_rtl = None
            best_offset_ltr = 0
            best_offset_rtl = 0

            for d in range(0, search_area):
                x_right = x - d
                if x_right - half_w >= 0:
                    # ltr
                    w_start_x = x_right - half_w
                    w_end_x = x_right + half_w + 1
                    w_right = gray_right[start_y:end_y, w_start_x:w_end_x]

                    score = score_fn(t_left, w_right)
                    if best_score_ltr is None or score < best_score_ltr:
                        best_score_ltr = score
                        best_offset_ltr = d

                x_left = x + d
                if x_left + half_w < w:
                    # rtl
                    w_start_x = x_left - half_w
                    w_end_x = x_left + half_w + 1
                    w_left = gray_left[start_y:end_y, w_start_x:w_end_x]

                    score = score_fn(t_right, w_left)
                    if best_score_rtl is None or score < best_score_rtl:
                        best_score_rtl = score
                        best_offset_rtl = d

            disparity_ltr[y, x] = best_offset_ltr
            disparity_rtl[y, x] = best_offset_rtl

    return disparity_ltr, disparity_rtl
