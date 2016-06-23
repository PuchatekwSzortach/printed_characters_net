"""
Module with sundry cv-related functions
"""

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

    return contour