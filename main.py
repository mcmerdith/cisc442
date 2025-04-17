import os
import cv2 as cv
from cv2.typing import MatLike
import numpy as np


def multi_resolution(image: MatLike, levels: int):
    pyr = [image]
    for _ in range(levels):
        h, w = image.shape[:2]
        blurred = cv.GaussianBlur(pyr[-1], (5, 5), 0)
        pyr.append(cv.resize(blurred, (w // 2, h // 2)))

    return pyr


def multi_scale(image: MatLike, levels: int):
    pyr = [image]
    for _ in range(levels):
        pyr.append(cv.GaussianBlur(pyr[-1], (5, 5), 0))

    return pyr


def laplacian(images: list[MatLike]):
    n = len(images)
    levels = []
    for i in range(n-1):
        h, w = images[i].shape[:2]
        levels.append(
            images[i].astype(np.float64) - cv.resize(images[i+1], (w, h))
        )
    levels.append(images[-1])

    return [level.clip(0, 255).astype(np.uint8) for level in levels]


def save_image(image: MatLike | list[MatLike], name: str):
    os.makedirs("output", exist_ok=True)
    if isinstance(image, list):
        for i, img in enumerate(image):
            cv.imwrite(f"output/{name}_{i}.png", img)
    else:
        cv.imwrite(f"output/{name}.png", image)


def main():
    image = cv.imread("Einstein.jpg")

    mr = multi_resolution(image.copy(), 3)
    ms = multi_scale(image.copy(), 3)
    lp = cv.Laplacian(image, cv.CV_64F).clip(0, 255).astype(np.uint8)
    ms_lp = laplacian(mr)
    mr_lp = laplacian(ms)

    save_image(image, "original")
    save_image(mr, "multi_resolution")
    save_image(ms, "multi_scale")
    save_image(lp, "laplacian")
    save_image(ms_lp, "multi_resolution_laplacian")
    save_image(mr_lp, "multi_scale_laplacian")


if __name__ == "__main__":
    main()
