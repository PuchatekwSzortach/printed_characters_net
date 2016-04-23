"""
Module with code for applying transformations to image.
These will be used to create artificial data from template images.
All functions transformation functions in this module use sensible default parameters based on input image size.
"""

import numpy as np
import cv2


def shift_image(image):

    padding = int(max(image.shape) / 5)
    large_image = 255 * np.ones((image.shape[0] + 2*padding, image.shape[1] + 2*padding)).astype(np.uint8)

    large_image[padding:image.shape[0] + padding, padding:image.shape[1] + padding] = image

    y_shift, x_shift = np.random.randint(-padding, padding, [2])

    return large_image[
           padding + y_shift:padding + y_shift + image.shape[0],
           padding + x_shift:padding + x_shift + image.shape[1]]

