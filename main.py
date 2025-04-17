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
            images[i] - cv.resize(images[i+1], (w, h))
        )
    levels.append(images[-1])

    return levels


def save_image(image: MatLike | list[MatLike], name: str):
    os.makedirs("output", exist_ok=True)
    if isinstance(image, list):
        for i, img in enumerate(image):
            cv.imwrite(f"output/{name}_{i}.png", img)
    else:
        cv.imwrite(f"output/{name}.png", image)


def normalize(image: MatLike | list[MatLike]):
    if isinstance(image, list):
        return [i.clip(0, 255).astype(np.uint8) for i in image]
    else:
        return image.clip(0, 255).astype(np.uint8)


def main():
    image = cv.imread("Einstein.jpg").astype(np.float64)

    mr = (multi_resolution(image.copy(), 3))
    ms = (multi_scale(image.copy(), 3))
    lp = (cv.Laplacian(image, cv.CV_64F))
    ms_lp = (laplacian(mr))
    mr_lp = (laplacian(ms))

    save_image(normalize(mr), "multi_resolution")
    save_image(normalize(ms), "multi_scale")
    save_image(normalize(lp), "laplacian")
    save_image(normalize(ms_lp), "multi_resolution_laplacian")
    save_image(normalize(mr_lp), "multi_scale_laplacian")


if __name__ == "__main__":
    main()
