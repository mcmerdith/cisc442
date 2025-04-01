import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from lib.util import gaussian_kernel_1d, save_image, show_image


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
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output


def reduce_image(I: MatLike):
    """
    Uniformly reduce the scale of an image by 1/2.

    Gaussian Smoothing -> Resizing (area interpolation)

    Args:
        I (MatLike): The input image in the shape (height, width, [channels])

    Returns:
        MatLike: The reduced image
    """

    gaussian_1d = gaussian_kernel_1d(5)

    vertical_gaussian = gaussian_1d.reshape(-1, 1)
    smoothed = convolve(I, vertical_gaussian)

    horizontal_gaussian = gaussian_1d.reshape(1, -1)
    smoothed = convolve(smoothed, horizontal_gaussian)

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


def align_images(I: MatLike, J: MatLike, points1: np.ndarray, points2: np.ndarray):
    """
    Align two images using the method described in the paper.

    Args:
        I (MatLike): The first image
        J (MatLike): The second image
        points1 (np.ndarray): The first image's matching points
        points2 (np.ndarray): The second image's matching points

    Returns:
        MatLike: The second image aligned to the first image
    """

    points1 = np.array(points1, dtype=np.float32)
    points2 = np.array(points2, dtype=np.float32)

    # Compute homography matrix
    H, mask = cv.findHomography(points2, points1, method=cv.RANSAC)

    # Warp img2 to img1's perspective
    height, width = I.shape[:2]
    aligned_img2 = cv.warpPerspective(J, H, (width, height))

    return aligned_img2


def compute_overlap_mask(img1, img2):
    """
    Computes a smooth mask for blending based on overlapping non-zero regions of two images.

    Args:
        img1: First image (base).
        img2: Second image (aligned to img1).

    Returns:
        Smooth transition mask (float32, single channel, range [0, 255]).
    """
    # Convert to grayscale if needed
    if img1.ndim == 3:
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    else:
        gray1 = img1
    if img2.ndim == 3:
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    else:
        gray2 = img2

    # Create binary masks where content exists
    mask1 = (gray1 > 10).astype(np.uint8)
    mask2 = (gray2 > 10).astype(np.uint8)

    return cv.cvtColor(np.where(mask1 & mask2, 0.5, 1), cv.COLOR_GRAY2BGR)


def mosaic_images(left: MatLike, right: MatLike, p1: np.ndarray = None, p2: np.ndarray = None):
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
        sift = cv.SIFT_create()
        flann = cv.FlannBasedMatcher(
            {"algorithm": 1, "trees": 5}, {"checks": 50})

        kp1, des1 = sift.detectAndCompute(left, None)
        kp2, des2 = sift.detectAndCompute(right, None)

        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        assert len(good) > 10, f"Not enough matches are found - {len(good)}/10"
        p1 = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        p2 = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # p1 = cv.cornerHarris(left, 2, 3, 0.04)
        # p2 = cv.cornerHarris(right, 2, 3, 0.04)

    M, mask = cv.findHomography(p1, p2, cv.RANSAC, 5.0)

    h, w = left.shape[:2]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
                     ).reshape(-1, 1, 2)

    output = np.hstack((left, cv.warpPerspective(right, M, (w, h))))

    # cv.resize(dst, (w, h), interpolation=cv.INTER_CUBIC)

    show_image(cv.drawMatches(left, kp1, right,
               kp2, good, None))
    # show_image(np.hstack((left, dst)))
    show_image(output)

    return output

    # img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    # aligned_img2 = align_images(left, right, p1, p2)

    # Step 4: Compute a mask automatically from overlap
    # mask = compute_overlap_mask(I, aligned_img2)
    mask = np.zeros_like(left)
    mask.fill(0.5)

    print(mask.shape)
    gp = [np.atleast_3d(gpl) for gpl in gaussian_pyramid(mask, 5)]
    lp1 = laplacian_pyramid(left, 5)
    lp2 = laplacian_pyramid(aligned_img2, 5)

    save_image(mask, "MASK.png")

    # for i in range(5):
    #     show_image(np.hstack((gp[i] * lp1[i], (1 - gp[i]) * lp2[i])))

    # return

    # Blend pyramids
    blended_pyramid = []
    for i in range(5):
        l = gp[i] * lp1[i] + (1 - gp[i]) * lp2[i]
        blended_pyramid.append(l)
        save_image(np.hstack((lp1[i], gp[i], lp2[i])), f"LP{i}.png")

    # Reconstruct image
    blended = reconstruct(blended_pyramid, 5)
    return np.clip(blended, 0, 255).astype(np.uint8)
