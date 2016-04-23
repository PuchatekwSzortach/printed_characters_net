"""
Module with code for applying transformations to image.
These will be used to create artificial data from template images.
All functions transformation functions in this module use sensible default parameters based on input image size.
"""

import numpy as np
import cv2


def shift_image(image):
    """
    Shift image by a random amount proportional to image size.
    :param image: image to be shifted
    :return: shifted image. Note that image will have black artifacts around corners. This is left
    in output on purpose, as should create a more difficult data set.
    """

    max_shift = int(max(image.shape) / 5)
    y_shift, x_shift = np.random.randint(-max_shift, max_shift, [2])

    transformation_matrix = np.eye(3)[:2, :]
    transformation_matrix[0, 2] = x_shift
    transformation_matrix[1, 2] = y_shift

    return cv2.warpAffine(image, transformation_matrix, dsize=image.shape)

