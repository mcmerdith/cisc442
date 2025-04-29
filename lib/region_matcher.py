import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from lib.stereo import stereo_analysis


def region_based(left_image: MatLike, right_image: MatLike, window_size: tuple[int, int], search_area: int, score_fn: str):
    gray_left = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)

    h, w = left_image.shape[:2]
    half_h, half_w = window_size[0] // 2, window_size[1] // 2

    kp = [(y, x) for y in range(half_h, h - half_h)
          for x in range(half_w, w - half_w)]

    return stereo_analysis(gray_left, gray_right, kp, window_size, search_area, score_fn)
