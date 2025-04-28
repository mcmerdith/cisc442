
from typing import Callable

import cv2 as cv
import numpy as np
from cv2.typing import MatLike


def normalize(image: MatLike):
    return cv.normalize(image, None, alpha=0,
                        beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)


def average_neighborhood(disparity: MatLike, max_size: int = 19):
    h, w = disparity.shape[:2]

    averaged = disparity.copy()

    base_size = 3
    # for y in range(half_base_size, h - half_base_size):
    #     for x in range(half_base_size, w - half_base_size):
    for y, x in np.argwhere(disparity == 0):
        # if np.abs(disparity[y, x]) > 0:
        #     continue
        # reset size
        size = base_size
        value = None
        while value is None:
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
            if non_zero.size < 5 and size + 2 < max_size:
                size += 2
            elif non_zero.size > 0:
                value = np.mean(non_zero).round().astype(averaged.dtype)
                # value = np.mean(neighborhood).round().astype(averaged.dtype)
            else:
                break
        if value is not None:
            averaged[y, x] = value

    return averaged


def validate(disp_left: MatLike, disp_right: MatLike, threshold: int = 0):
    """
    Perform left-right consistency check on disparity maps.

    Args:
        disp_left (np.ndarray): Left-to-right disparity map.
        disp_right (np.ndarray): Right-to-left disparity map.
        threshold (int): Allowed difference between disparities.

    Returns:
        valid_disp (np.ndarray): Disparity map after validity checking.
    """
    height, width = disp_left.shape
    valid_disp = disp_left.copy()

    for y in range(height):
        for x in range(width):
            d = disp_left[y, x]
            if d == 0:
                continue  # Already a gap

            x_r = x - int(d)
            if x_r >= 0 and x_r < width:
                d_prime = disp_right[y, x_r]
                if np.abs(int(d) - int(d_prime)) > threshold:
                    valid_disp[y, x] = 0  # Invalidate
            else:
                valid_disp[y, x] = 0  # Out of bounds → invalidate

    return valid_disp


def kp_overlay(image: MatLike, kp):
    return cv.drawKeypoints(image.copy(), kp, None)


def overlay(image: MatLike, points: np.ndarray):
    overlay = np.zeros(image.shape[:2], dtype=np.uint8)
    overlay[points[:, 1], points[:, 0]] = 1
    overlay = cv.dilate(overlay, None, iterations=2)

    overlayed = image.copy()
    overlayed[overlay == 1] = [0, 0, 255]

    return overlayed


def draw_matches(left_image: MatLike, left_kp: np.ndarray, right_image: MatLike, right_kp: np.ndarray):
    return cv.drawMatches(left_image, left_kp, right_image, right_kp, None, None)


def harris(image: MatLike):
    """
    Feature detection using Harris corner detector

    Returns a binary image of the corners
    """
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    corners = cv.cornerHarris(gray_image, 2, 3, 0.04)
    ret, corners = cv.threshold(corners, 0.01*corners.max(), 255, 0)
    corners = np.uint8(corners)

    ret, labels, stats, centroids = cv.connectedComponentsWithStats(corners)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    detections = np.int32(
        cv.cornerSubPix(gray_image, np.float32(
            centroids), (5, 5), (-1, -1), criteria)
    )

    return detections


ScoreFunction = Callable[[MatLike, MatLike], float]


def score_SAD(first: MatLike, second: MatLike):
    first = first.astype(np.int32)
    second = second.astype(np.int32)
    return np.sum(np.abs(first - second))


def score_SSD(first: MatLike, second: MatLike):
    first = first.astype(np.int32)
    second = second.astype(np.int32)
    return np.sum(np.square(first - second))


def score_NCC(first: MatLike, second: MatLike):
    first = first.astype(np.int32)
    second = second.astype(np.int32)
    # questionable...
    m1 = np.mean(first)
    m2 = np.mean(second)

    numerator = np.sum((first - m1) * (second - m2))
    denominator = np.sqrt(np.sum(np.square(first - m1)) *
                          np.sum(np.square(second - m2)))

    if denominator == 0:
        return 1

    return -(numerator / denominator)
