from typing import Callable
import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from numba import njit

ScoreFunction = Callable[[MatLike, MatLike], np.int32 | np.float64]
"""Returns a score for the match between two images. Lower is better."""


@njit
def score_SAD(first: MatLike, second: MatLike) -> np.float64:
    first = first.astype(np.int32)
    second = second.astype(np.int32)
    return np.sum(np.abs(first - second), dtype=np.float64)


@njit
def score_SSD(first: MatLike, second: MatLike) -> np.float64:
    first = first.astype(np.int32)
    second = second.astype(np.int32)
    return np.sum(np.square(first - second), dtype=np.float64)


@njit
def score_NCC(first: MatLike, second: MatLike) -> np.float64:
    """
    Inverted normalized cross correlation


    """
    first = first.astype(np.int32)
    second = second.astype(np.int32)

    m1 = np.mean(first)
    m2 = np.mean(second)

    centered_first = first - m1
    centered_second = second - m2

    numerator = np.sum(centered_first * centered_second)
    denominator = np.sqrt(np.sum(np.square(centered_first)) *
                          np.sum(np.square(centered_second)))

    if denominator == 0:
        return np.inf

    return -(numerator / denominator)


@njit
def calculate_difference(score_fn: str, first: MatLike, second: MatLike) -> np.float64:
    if score_fn == "sad":
        return score_SAD(first, second)
    elif score_fn == "ssd":
        return score_SSD(first, second)
    elif score_fn == "ncc":
        return score_NCC(first, second)
    else:
        raise ValueError(f"Invalid score function: {score_fn}")


@njit
def stereo_analysis(left_image: MatLike,
                    right_image: MatLike,
                    left_kp: np.ndarray,
                    window_size: tuple[int, int], search_area: int, score_fn: str):
    """
    Compute disparity map using stereo analysis

    Parameters
    ----------
    left_image : MatLike
        Left image
    right_image : MatLike
        Right image
    left_kp : np.ndarray
        (y,x) keypoints in left image
    window_size : tuple[int, int]
        (height, width) of window
    search_area : int
        Number of pixels to search around each keypoint
    score_fn : ScoreFunction
        Function to use for matching

    Returns
    -------
    disparity : np.ndarray
        Disparity map
    """

    h, w = left_image.shape[:2]
    half_h, half_w = window_size[0] // 2, window_size[1] // 2

    disp_left = np.zeros((h, w), dtype=left_image.dtype)
    disp_right = np.zeros((h, w), dtype=right_image.dtype)

    for (y_t, x_t) in left_kp:
        y_t_start = y_t - half_h
        y_t_end = y_t + half_h + 1
        x_t_start = x_t - half_w
        x_t_end = x_t + half_w + 1

        if y_t_start < 0 or y_t_end > h or x_t_start < 0 or x_t_end > w:
            continue

        t_left = left_image[y_t_start:y_t_end, x_t_start:x_t_end]
        t_right = right_image[y_t_start:y_t_end, x_t_start:x_t_end]

        best_score_ltr = None
        best_offset_ltr = None
        best_score_rtl = None
        best_offset_rtl = None

        for d in range(0, search_area):
            x_w_left = x_t - d
            if x_w_left - half_w >= 0:
                # ltr
                x_w_start = x_w_left - half_w
                x_w_end = x_w_left + half_w + 1
                w_right = right_image[y_t_start:y_t_end, x_w_start:x_w_end]

                score = calculate_difference(score_fn, t_left, w_right)
                if best_score_ltr is None or score < best_score_ltr:
                    best_score_ltr = score
                    best_offset_ltr = d

            x_w_right = x_t + d
            if x_w_right + half_w < w:
                # rtl
                x_w_start = x_w_right - half_w
                x_w_end = x_w_right + half_w + 1
                w_left = left_image[y_t_start:y_t_end, x_w_start:x_w_end]

                score = calculate_difference(score_fn, t_right, w_left)
                if best_score_rtl is None or score < best_score_rtl:
                    best_score_rtl = score
                    best_offset_rtl = d

        if best_offset_ltr is not None:
            disp_left[y_t, x_t] = best_offset_ltr

        if best_offset_rtl is not None:
            disp_right[y_t, x_t] = best_offset_rtl

    return disp_left, disp_right
