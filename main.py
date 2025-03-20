import cv2 as cv
import numpy as np
from os import path, makedirs

OUT_DIR = "output"

def detect(image: cv.typing.MatLike, filename: str):
  image = image.copy()
  canny =  cv.Canny(image, threshold1=100, threshold2=200)

  gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  gray = np.float32(gray)

  harris = cv.cornerHarris(gray, 2, 3, 0.04)
  harris = cv.dilate(harris, None)
  image[harris>0.01*harris.max()]=[0,0,0]

  cv.imwrite(path.join(OUT_DIR, f"{filename}_edges.jpg"), canny)
  cv.imwrite(path.join(OUT_DIR, f"{filename}_corners.jpg"), image)

def main():
  makedirs(OUT_DIR, exist_ok=True)

  image = cv.imread("flower_HW4.jpg")
  (h, w) = image.shape[:2]

  # base image
  detect(image, "base")

  # rotated
  rotation = cv.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), 60, 1)
  rotated = cv.warpAffine(image.copy(), rotation, (image.shape[1], image.shape[0]))

  detect(rotated, "rotated")

  # scaled
  scale = cv.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), 0, 1.6)
  scaled = cv.warpAffine(image.copy(), scale, (image.shape[1], image.shape[0]))

  detect(scaled, "scaled")

  # sheared x
  shear_x = np.float32([[1, 1.2, -w/2], 
                      [0, 1, 0]])
  sheared_x = cv.warpAffine(image.copy(), shear_x, (image.shape[1], image.shape[0]))

  detect(sheared_x, "sheared_x")

  # sheared y
  shear_y = np.float32([[1, 0, 0], 
                      [1.4, 1, -h/2]])
  sheared_y = cv.warpAffine(image.copy(), shear_y, (image.shape[1], image.shape[0]))

  detect(sheared_y, "sheared_y")

  cv.imwrite("temp.jpg", sheared_x)

if __name__ == '__main__':
  main()