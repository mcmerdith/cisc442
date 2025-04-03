import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from lib.util import gaussian_kernel_1d, load_kernel, save_image
from lib.gui import ShowImageGui


def convolve(I: MatLike, H: MatLike, mode='reflect') -> MatLike:
    """
    Perform convolution of an image with a given kernel.

    Args:
        I (MatLike): The input image in the shape (height, width, channels)
        H (MatLike): The kernel in the shape (height, width)
        mode (str): Mode for np.pad()

    Returns:
        MatLike: The result of the convolution
    """

    return cv.filter2D(I, -1, H)

    # accomodate grayscale and color images
    is_color = I.ndim == 3

    # get dimensions
    img_h, img_w = I.shape[:2]
    kernel_h, kernel_w = H.shape

    # compute required padding
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    # pad the image
    if is_color:
        # no padding for color channels
        padding = ((pad_h, pad_h), (pad_w, pad_w), (0, 0))
    else:
        padding = ((pad_h, pad_h), (pad_w, pad_w))

    # pad the image
    I_padded = np.pad(I, padding, mode=mode)

    # create a blank output image
    output = np.zeros(I.shape, dtype=np.float32)

    # do the convolution for each channel of each pixel
    for y in range(img_h):
        for x in range(img_w):
            if is_color:
                # apply to all channels for RGB
                for channel in range(I.shape[2]):
                    output[y, x, channel] = np.sum(
                        H * I_padded[y:y + kernel_h, x:x + kernel_w, channel]
                    )
            else:
                # apply to grayscale
                output[y, x] = np.sum(
                    H * I_padded[y:y + kernel_h, x:x + kernel_w]
                )

    # clip to valid range [0, 255] and cast to uint8
    # output = np.clip(output, 0, 255).astype(np.uint8)

    return output


def gaussian_blur(I: MatLike, sigma: float = 1.0):
    """
    Apply a Gaussian blur to an image.

    Args:
        I (MatLike): The input image in the shape (height, width, [channels])
        sigma (float): The standard deviation of the Gaussian kernel

    Returns:
        MatLike: The result of the convolution
    """

    gaussian_1d = gaussian_kernel_1d(5, sigma=sigma)

    vertical_gaussian = gaussian_1d.reshape(-1, 1)
    smoothed = convolve(I, vertical_gaussian)

    horizontal_gaussian = gaussian_1d.reshape(1, -1)
    smoothed = convolve(smoothed, horizontal_gaussian)

    return smoothed


def reduce_image(I: MatLike):
    """
    Uniformly reduce the scale of an image by 1/2.

    Gaussian Smoothing -> Resizing (area interpolation)

    Args:
        I (MatLike): The input image in the shape (height, width, [channels])

    Returns:
        MatLike: The reduced image
    """

    smoothed = gaussian_blur(I)

    reduced = cv.resize(
        smoothed, (I.shape[1] // 2, I.shape[0] // 2), interpolation=cv.INTER_AREA
    )

    return reduced


def expand_image(I: MatLike, shape: tuple[int, int] | None = None):
    """
    Uniformly expand the scale of an image by 2 using cubic interpolation.

    Args:
        I (MatLike): The input image in the shape (height, width, [channels])

    Returns:
        MatLike: The expanded image
    """

    if shape is None:
        # default is double the size
        shape = (I.shape[0]*2, I.shape[1]*2)
    else:
        assert shape[0] > I.shape[0] and \
            shape[1] > I.shape[1], f"Invalid expansion shape (too small) {shape} < {I.shape}"

    return cv.resize(I, (shape[1], shape[0]), interpolation=cv.INTER_CUBIC)


def gaussian_pyramid(I: MatLike, n: int):
    """
    Compute the n-level Gaussian pyramid of an image.

    Args:
        I (MatLike): The input image
        n (int): The number of levels

    Returns:
        list[MatLike]: The n-level Gaussian pyramid
    """

    levels = [I]
    for _ in range(n-1):
        levels.append(reduce_image(levels[-1]))

    return levels


def laplacian_pyramid(I: MatLike, n):
    """
    Compute the n-level Laplacian pyramid of an image.

    Args:
        I (MatLike): The input image
        n (int): The number of levels

    Returns:
        list[MatLike]: The n-level Laplacian pyramid
    """

    pyramid = gaussian_pyramid(I, n)

    levels = []
    for i in range(n-1):
        levels.append(
            pyramid[i] - expand_image(pyramid[i+1], shape=pyramid[i].shape))
    levels.append(pyramid[-1])

    return levels


def reconstruct(LI: list[MatLike], n):
    """
    Reconstruct an image from a Laplacian pyramid.

    Args:
        LI (list[MatLike]): The Laplacian pyramid
        n (int): The number of levels

    Returns:
        MatLike: The reconstructed image
    """
    assert len(LI) == n

    current_level = LI[-1]
    for i in reversed(range(n-1)):
        current_level = LI[i] + \
            expand_image(current_level, shape=LI[i].shape)

    return current_level


def match_images(left: MatLike, right: MatLike):
    """
    Find the matching points between two images.

    Args:
        left (MatLike): The left image
        right (MatLike): The right image

    Returns:
        np.ndarray: The left image's matching points
    """

    # preprocess with canny
    g_left = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
    g_right = cv.cvtColor(right, cv.COLOR_BGR2GRAY)
    proc_left = g_left
    proc_right = g_right
    # proc_left = laplacian_pyramid(g_left, 3)[-2]
    # proc_right = laplacian_pyramid(g_right, 3)[-2]
    # proc_left = cv.Canny(gaussian_blur(g_left), 100, 200)
    # proc_right = cv.Canny(gaussian_blur(g_right), 100, 200)

    ShowImageGui(image=[proc_left, proc_right]).init()

    sift = cv.SIFT_create()
    flann = cv.FlannBasedMatcher(
        {"algorithm": 1, "trees": 5})
    # orb = cv.ORB_create()
    # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    kp1, des1 = sift.detectAndCompute(proc_left, None)
    kp2, des2 = sift.detectAndCompute(proc_right, None)
    # kp1, des1 = orb.detectAndCompute(left, None)
    # kp2, des2 = orb.detectAndCompute(right, None)

    matches = flann.knnMatch(des1, des2, k=2)
    # matches = bf.match(des1, des2)

    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    # good = sorted(good, key=lambda m: np.abs(m.distance-n.distance))[:5]

    ShowImageGui(image=cv.drawMatches(
        proc_left, kp1, proc_right, kp2, good, None)).init()

    assert len(good) > 4, f"Not enough matches are found - {len(good)}/4"
    p1 = np.float32(
        [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    p2 = np.float32(
        [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return p1, p2


def align_image(left: MatLike, right: MatLike, p1: np.ndarray, p2: np.ndarray):
    """
    Align the right image to the left image.

    Args:
        left (MatLike): The left image
        right (MatLike): The right image
        p1 (np.ndarray): The left image's matching points
        p2 (np.ndarray): The right image's matching points

    Returns:
        MatLike: The right image aligned to the left image
    """

    M, mask = cv.findHomography(
        p2.astype(np.float32), p1.astype(np.float32), cv.LMEDS, 5.0)
    # M, mask = cv.estimateAffine2D(p2.astype(np.float32), p1.astype(
    #     np.float32), method=cv.RANSAC, ransacReprojThreshold=5.0)
    matchesMask = mask.ravel().tolist()

    assert M is not None, f"No homography found"

    lh, lw = left.shape[:2]

    h_right, w_right = right.shape[:2]
    corners = np.array([
        [0, 0],
        [w_right, 0],
        [w_right, h_right],
        [0, h_right]
    ], dtype=np.float32).reshape(-1, 1, 2)

    warped_corners = cv.perspectiveTransform(corners, M)
    x_min = np.floor(warped_corners[:, 0, 0].min()).astype(int)
    x_max = np.ceil(warped_corners[:, 0, 0].max()).astype(int)

    # x_offset = -x_min if x_min < 0 else 0

    # T = np.array([
    #     [1, 0, x_offset],
    #     [0, 1, 0],
    #     [0, 0, 1]
    # ], dtype=np.float32)

    # H_adjusted = T @ M

    aligned = cv.warpPerspective(right, M, (x_max, lh))
    # aligned = cv.warpAffine(right, M, (w, h))

    return aligned


def mosaic_images(left: MatLike, right: MatLike, p1: np.ndarray = None, p2: np.ndarray = None, n=3):
    """
    Mosaic two images together.

    Args:
        I (MatLike): The first image
        J (MatLike): The second image
        p1 (np.ndarray): The first image's matching points
        p2 (np.ndarray): The second image's matching points

    Returns:
        MatLike: The mosaiced image
    """

    if p1 is None or p2 is None:
        p1, p2 = match_images(left, right)

    aligned_right = align_image(left, right, p1, p2)
    # blended = cv.addWeighted(left, 0.5, aligned_right, 0.5, 0)

    # ShowImageGui(
    #     image=[left, aligned_right, right]).init()

    # return aligned_right
    lh, lw = left.shape[:2]

    canvas = np.ones_like(aligned_right)
    canvas[:, :left.shape[1]] = left

    overlap_start = lw - 50
    overlap_end = lw

    # compute blend mask
    mask = np.zeros_like(canvas, dtype=np.float32)
    mask[:, :overlap_start] = 1
    # mask falloff
    alpha = np.linspace(1, 0, overlap_end -
                        overlap_start)[None, :, None]
    # match image height
    alpha = np.repeat(alpha, left.shape[0], axis=0)
    # mask[:, :overlap_end] = alpha
    alpha = np.repeat(alpha, 3, axis=2)  # (H, W, 3)
    mask[:, overlap_start:overlap_end, :] = alpha

    # pyramid time
    gp = gaussian_pyramid(mask, n)
    lp1 = laplacian_pyramid(canvas, n)
    lp2 = laplacian_pyramid(aligned_right, n)

    # ShowImageGui(image=gp).init()

    # Blend pyramids
    blended_pyramid = []
    for i in range(n):
        l = (gp[i] * lp1[i]) + (1 - gp[i]) * lp2[i]
        blended_pyramid.append(l)

    ShowImageGui(image=blended_pyramid).init()

    # Reconstruct image
    blended = reconstruct(blended_pyramid, n).clip(0, 255).astype(np.uint8)
    ShowImageGui(image=[blended, canvas, aligned_right]).init()
    return np.clip(blended, 0, 255).astype(np.uint8)
