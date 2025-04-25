import os
import cv2 as cv
import numpy as np


method = "region"
left_image = cv.imread(left_image_path)
right_image = cv.imread(right_image_path)


def average_neighborhood(disparity):
    pass


def region_based(left_image, right_image, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y, disparity):
    pass


def feature_based(left_image, right_image, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y, disparity):
    pass


input('Enter distance [SAD, SSD, NCC]:'.format(i))
input('Enter method [region, feature]:'.format(i))
int(input('Enter search range (need to be integer):'.format(i)))
int(input('Enter template_x_size (need to be odd integer):'.format(i)))
int(input('Enter template_y_size (need to be odd integer):'.format(i)))


if method == 'region':
    disparity = region_based()
elif method == 'feature':
    disparity = feature_based()

for i in range(2):
    disparity = average_neighborhood(disparity)

disparity = cv.normalize(disparity, None, alpha=0,
                         beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
cv.imwrite('disparity.png', disparity)
