import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from numba import njit


def normalize(image: MatLike):
    return cv.normalize(image, None, alpha=0,
                        beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)


# @njit
def average_neighborhood(disparity: MatLike, max_size: int = 21, minimum_neighbors: int = 5) -> MatLike:
    """
    Fill in gaps in disparity map using the average of it's non-zero neighbors

    Args:
        disparity (Matlike): Disparity map
        max_size (int): Maximum size of neighborhood to consider

    Returns:
        np.ndarray: Disparity map with gaps filled in
    """
    h, w = disparity.shape[:2]

    averaged = disparity.copy()

    base_size = 3
    for y, x in np.argwhere(disparity == 0):
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

            # non_zero = np.argwhere(neighborhood)
            non_zero = np.where(neighborhood != 0)

            if non_zero.shape[0] > minimum_neighbors:
                # A = []
                # B = []
                # for yy, xx in non_zero:
                #     dx = xx - x
                #     dy = yy - y
                #     A.append([dx*dx, dy*dy, dx*dy, dx, dy, 1])
                #     B.append(neighborhood[yy, xx])
                # A = np.array(A, dtype=np.float64)
                # B = np.array(B, dtype=np.float64)
                # coeff, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
                # averaged[y, x] = coeff[-1]
                averaged[y, x] = \
                    np.mean(non_zero).round().astype(
                        averaged.dtype)
            else:
                # not enough neighbors
                if size + 2 < max_size:
                    # try again with larger size
                    size += 2
                    continue
            break

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
                valid_disp[y, x] = 0  # Out of bounds

    return valid_disp
