import cv2 as cv
from cv2.typing import MatLike


class PointSelector:
    def __init__(self, image: MatLike):
        self.points = []
        self.window_name = "Select points"
        self.image = image

    def show(self):
        cv.imshow(self.window_name, self.image)
        cv.setMouseCallback(self.window_name, self.mouse_click)
        cv.waitKey(0)

    def mouse_click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.image = cv.circle(self.image, (x, y), 5, (0, 0, 255), -1)
            cv.imshow(self.window_name, self.image)
