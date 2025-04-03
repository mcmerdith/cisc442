from dataclasses import dataclass
from time import time
import cv2 as cv
from cv2.typing import MatLike
from lib.util import logger, normalize_u8

QUIT = "q"
PROCEED = " "
UNDO = "u"
REDO = "z"


@dataclass(kw_only=True)
class GuiWindow:
    name: str
    timeout: int = None

    def __post_init__(self):
        self.defer = False
        self.windows = None

    def init(self):
        self.destroyed = False
        self.show()

        if self.windows:
            for i, window in enumerate(self.windows):
                cv.setMouseCallback(window, lambda *args,
                                    idx=i: self.handle_click(*args, idx=idx))
        else:
            cv.setMouseCallback(self.name, self.handle_click)

        if not self.defer:
            self.run()

        return self

    def show(self):
        raise NotImplementedError

    def destroy(self):
        if isinstance(self.windows, list):
            for window in self.windows:
                cv.destroyWindow(window)
        else:
            cv.destroyWindow(self.name)
        self.destroyed = True

    def is_closed(self):
        if isinstance(self.windows, list):
            return self.destroyed or all([cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) < 1 for window in self.windows])
        else:
            return self.destroyed or cv.getWindowProperty(self.name, cv.WND_PROP_VISIBLE) < 1

    def handle_key(self, key: str):
        if key == PROCEED:
            self.destroy()
        elif key == QUIT:
            logger.info("User requested exit")
            exit(0)

    def handle_click(self, event: int, x: int, y: int, flags: int, param, *, idx: int = None):
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
    name: str = "Select points"
    images: list[MatLike]

    def __post_init__(self):
        super().__post_init__()
        self.images = [image.copy() for image in self.images]
        self.windows = [self.name + " (left)", self.name + " (right)"]
        self.points = [[], []]

    # def init(self):
    #     self.defer = True
    #     super().init()
    #     cv.setMouseCallback(self.windows[0],
    #                         lambda *args: self.handle_click(*args, idx=0))
    #     cv.setMouseCallback(self.windows[1],
    #                         lambda *args: self.handle_click(*args, idx=1))
    #     self.run()
    #     return self

    def show(self):
        cv.imshow(self.windows[0], self.images[0])
        cv.imshow(self.windows[1], self.images[1])

    def handle_key(self, key: str):
        # safeguard proceeding without the right number of points
        if key == PROCEED and len(self.points[0]) != len(self.points[1]):
            return
        super().handle_key(key)

    def handle_click(self, event, x, y, flags, param, *, idx: int):
        if event == cv.EVENT_LBUTTONDOWN:
            # safeguard adding a point without a match in the other image
            if len(self.points[idx]) > len(self.points[1 - idx]):
                return

            self.points[idx].append((x, y))
            self.images[idx] = cv.circle(
                self.images[idx], (x, y), 5, (255 - (255 / len(self.points[idx])), 0, 255 / len(self.points[idx])), -1)
            self.show()


@dataclass(kw_only=True)
class ShowImageGui(GuiWindow):
    name: str = "Preview"
    image: MatLike | list[MatLike]

    def __post_init__(self):
        super().__post_init__()
        self.index = 0
        if not isinstance(self.image, list):
            self.image = [self.image]

    def show(self):
        cv.imshow(self.name, normalize_u8(self.image[self.index]))
        cv.setWindowTitle(
            self.name, f"{self.name} ({self.index+1}/{len(self.image)})")

    def handle_key(self, key: str):
        super().handle_key(key)
        if key == "l":
            self.index = (self.index + 1) % len(self.image)
            self.show()
        if key == "h":
            self.index = (self.index - 1) % len(self.image)
            self.show()
