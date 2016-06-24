"""
Module with sundry cv-related functions
"""

import numpy as np

def get_ordered_card_contour(contour):
    """
    Given a 4-points contour, return a version that has left top point as first element,
    and then proceeds clockwise.
    :param contour: A 4-points, roughly rectangular contour that represents a card candidate
    :return: contour with points rotated so that first contour is top left, and following
    contours are in clockwise-order
    """

    # A sanity check
    if len(contour) != 4:
        raise ValueError("Contour length must be 4")

    ordered_contour = np.zeros(shape=(4,2))

    # Sum coordinates for each point
    sums = np.sum(contour, axis=1)

    # Top left contour will have smallest coordinates sum,
    # right bottom contour will have largest coordinates sum
    ordered_contour[0] = contour[np.argmin(sums)]
    ordered_contour[2] = contour[np.argmax(sums)]

    differences = np.diff(contour, axis=1)

    # Top right contour will have smallest coordinates difference,
    # bottom left contour will have largest coordinates difference
    ordered_contour[1] = contour[np.argmin(differences)]
    ordered_contour[3] = contour[np.argmax(differences)]

    return ordered_contour
