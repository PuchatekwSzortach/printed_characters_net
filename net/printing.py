"""
Utilities related to printing templates to pdf files
"""

import numpy as np


def get_image_with_border(image, border_width):
    """
    Add a border around image
    :param image: image
    :param border_width: border width
    :return: image with black border around it
    """
    height, width = image.shape
    bordered_shape = ((2 * border_width) + height, (2 * border_width) + width)
    bordered_image = np.zeros(bordered_shape)

    bordered_image[border_width: height + border_width, border_width: width + border_width] = image
    return bordered_image