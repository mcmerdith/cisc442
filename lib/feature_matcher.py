import cv2 as cv
import numpy as np

from cv2.typing import MatLike

from lib.common import show_image
from lib.stereo import stereo_analysis


def feature_based(left_image: MatLike, right_image: MatLike, window_size: tuple[int, int], search_area: int, score_fn: str, *, threshold: float = 0.001):
    gray_left = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)

    h, w = gray_left.shape[:2]

    harris_img = cv.resize(
        gray_left, (w // 2, h // 2), interpolation=cv.INTER_AREA)

    harris_response = cv.cornerHarris(harris_img, 2, 3, 0.04)

    harris_response[harris_response < threshold*harris_response.max()] = 0
    harris_response[harris_response > threshold*harris_response.max()] = 255

    harris_response = cv.resize(
        harris_response, (w, h), interpolation=cv.INTER_CUBIC)
    harris_kp = np.argwhere(harris_response)

    disparity_ltr, disparity_rtl = stereo_analysis(gray_left, gray_right, harris_kp,
                                                   window_size, search_area, score_fn)

    disparity_ltr = cv.dilate(disparity_ltr, None)
    disparity_rtl = cv.dilate(disparity_rtl, None)

    return disparity_ltr, disparity_rtl
