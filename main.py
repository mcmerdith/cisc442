import cv2 as cv
import numpy as np

from cv2.typing import MatLike

from os import listdir, makedirs, path


def show_image(image: MatLike):
    cv.imshow("image", image)
    while True:
        key = cv.waitKey(100)
        # space or escape
        if key == 32 or key == 27:
            break
        if cv.getWindowProperty("image", cv.WND_PROP_VISIBLE) < 1:
            break
    cv.destroyWindow("image")


def load_image(filename: str):
    return cv.imread(path.join("input", filename))


def save_image(filename: str, image: MatLike):
    makedirs("output", exist_ok=True)
    cv.imwrite(path.join("output", filename), image)


def kp_overlay(image: MatLike, kp):
    return cv.drawKeypoints(image.copy(), kp, None)


def overlay(image: MatLike, points: np.ndarray):
    overlay = np.zeros(image.shape[:2], dtype=np.uint8)
    overlay[points[:, 1], points[:, 0]] = 1
    overlay = cv.dilate(overlay, None, iterations=2)

    overlayed = image.copy()
    overlayed[overlay == 1] = [0, 0, 255]

    return overlayed


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

    return overlay(image, detections), detections


def sift(image: MatLike):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray_image, None)
    return kp_overlay(image, kp), kp, des


def shi_tomasi(image: MatLike):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    corners = np.int32(
        [i.ravel() for i in cv.goodFeaturesToTrack(gray_image, 500, 0.01, 10)]
    )
    return overlay(image, corners), np.int32(corners)


def fast(image: MatLike):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    fast = cv.FastFeatureDetector_create()
    kp = fast.detect(gray_image, None)
    return kp_overlay(image, kp), kp


def orb(image: MatLike):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create()
    kp, des = orb.detectAndCompute(gray_image, None)
    return kp_overlay(image, kp), kp, des


def main():
    images = [(filename, load_image(filename))
              for filename in listdir("input")]
    detectors = [harris, sift, shi_tomasi, fast, orb]

    for filename, image in images:
        for detector in detectors:
            overlayed, *_ = detector(image)
            save_image(f"{filename}_{detector.__name__}.png", overlayed)

    pass


if __name__ == "__main__":
    main()
