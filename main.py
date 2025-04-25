import cv2 as cv
import numpy as np

from cv2.typing import MatLike

from os import listdir, path


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


def harris(image: MatLike) -> np.ndarray:
    image = cv.cvtColor(image.copy(), cv.COLOR_BGR2GRAY)
    show_image(image)
    pass


def main():
    images = [cv.imread(path.join("input", filename))
              for filename in listdir("input")]

    harris(images[0])

    pass


if __name__ == "__main__":
    main()
