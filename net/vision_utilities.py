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

    all_indices = set(range(4))

    top_points_indices = set(index for index, point in enumerate(contour)
                          if is_point_above_region_midpoint(contour, point))

    bottom_points_indices = all_indices.difference(top_points_indices)

    left_points_indices = set(index for index, point in enumerate(contour)
                           if is_point_to_the_left_of_region_midpoint(contour, point))

    right_points_indices = all_indices.difference(left_points_indices)

    left_top_index = top_points_indices.intersection(left_points_indices).pop()
    right_top_index = top_points_indices.intersection(right_points_indices).pop()

    right_bottom_index = bottom_points_indices.intersection(right_points_indices).pop()
    left_bottom_index = bottom_points_indices.intersection(left_points_indices).pop()

    ordered_indices = [left_top_index, right_top_index, right_bottom_index, left_bottom_index]

    ordered_contour = np.zeros(shape=(4,2))

    for index in range(4):
        ordered_contour[index] = contour[ordered_indices[index]]

    return ordered_contour


def is_point_above_region_midpoint(contour, point):
    """
    Given a contour of a region and a point, return true if point is above
    contours midpoint
    :param contour: 2D numpy array of (x,y) coordinates
    :param point: a single coordinate
    :return: True if point is above middle point of region, False otherwise.
    Please not that we consider a point A is considered to be above point B if it has a lower
    y-coordinate, since
    """

    mean_y_coordinate = np.mean(contour[:, 1])
    return point[1] < mean_y_coordinate


def is_point_to_the_left_of_region_midpoint(contour, point):
    """
    Given a contour of a region and a point, return true if point is to the left of
    contours midpoint
    :param contour: 2D numpy array of (x,y) coordinates
    :param point: a single coordinate
    :return: True if point is to the left of middle point of region, False otherwise.
    """

    mean_x_coordinate = np.mean(contour[:, 0])
    return point[0] < mean_x_coordinate