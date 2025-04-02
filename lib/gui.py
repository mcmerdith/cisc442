from dataclasses import dataclass
from time import time
import cv2 as cv
from cv2.typing import MatLike
from lib.util import logger

QUIT = "q"
PROCEED = " "
UNDO = "u"
REDO = "z"


@dataclass(kw_only=True)
class GuiWindow:
    name: str
    timeout: int = None

    def __post_init__(self):
        pass

    def init(self):
        self.destroyed = False
        self.show()
        if not self.defer:
            cv.setMouseCallback(self.name, self.handle_click)
            self.run()

        return self

    def show(self):
        raise NotImplementedError

    def destroy(self):
        cv.destroyWindow(self.name)
        self.destroyed = True

    def is_closed(self):
        return self.destroyed or cv.getWindowProperty(self.name, cv.WND_PROP_VISIBLE) < 1

    def handle_key(self, key: str):
        if key == PROCEED:
            self.destroy()
        elif key == QUIT:
            logger.info("User requested exit")
            self.destroy()
            exit(0)

    def handle_click(self, event, x, y, flags, param):
        pass

    def run(self):
        opened = time()
        while True:
            # Exit if window is closed
            if self.is_closed():
                break

            key = cv.waitKey(100)
            try:
                self.handle_key(chr(key))
            except:
                pass  # invalid or no key

            if self.timeout is not None and time() - opened > (self.timeout / 1000):
                # window is now too old
                self.destroy()


@dataclass(kw_only=True)
class PointMatcherGui(GuiWindow):
    name = "Select points"
    image: tuple[MatLike, MatLike]

    def __post_init__(self):
        self.defer = True
        self.window1_name = self.name + " (left)"
        self.window2 = self.name + " (right)"
        self.points = ([], [])

    def init(self):
        super().init()
        # setup handlers for both windows
        cv.setMouseCallback(self.window1_name,
                            lambda *args: self.handle_click(*args, idx=0))
        cv.setMouseCallback(self.window2_name,
                            lambda *args: self.handle_click(*args, idx=1))
        # run the main loop
        self.run()
        return self

    def show(self):
        cv.imshow(self.window1_name, self.image[0])
        cv.imshow(self.window2, self.image[1])

    def handle_key(self, key: str):
        # safeguard proceeding without the right number of points
        if key == PROCEED and len(self.points[0]) != len(self.points[1]):
            return
        super().handle_key(key)

    def handle_click(self, event, x, y, flags, param, *, idx: int):
        if event == cv.EVENT_LBUTTONDOWN:
            # safeguard adding a point without a match in the other image
            if len(self.points[0]) != len(self.points[1]):
                return

            self.points[idx].append((x, y))
            self.image[idx] = cv.circle(
                self.image[idx], (x, y), 5, (0, 0, 255 / len(self.points[idx])), -1)
            self.show()


@dataclass(kw_only=True)
class ShowImageGui(GuiWindow):
    name: str = "Preview"
    image: MatLike | list[MatLike]

    def __post_init__(self):
        self.index = 0
        if not isinstance(self.image, list):
            self.image = [self.image]
        if len(self.image) > 1:
            self.name = f"{self.name} ({len(self.image)} images)"

    def show(self):
        cv.imshow(self.name, self.image[self.index])

    def handle_key(self, key: str):
        super().handle_key(key)
        if key == "l":
            self.index = (self.index + 1) % len(self.image)
            self.show()
        if key == "h":
            self.index = (self.index - 1) % len(self.image)
            self.show()
