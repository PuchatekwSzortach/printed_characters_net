"""
Tests for vision_utilities module
"""
import numpy as np
import pytest

import net.vision_utilities

def test_is_point_above_region_midpoint_1():

    contour = np.array(
        [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10]
        ])

    assert True == net.vision_utilities.is_point_above_region_midpoint(contour, [10, 0])
    assert False == net.vision_utilities.is_point_above_region_midpoint(contour, [10, 10])

# def test_get_ordered_card_contour_throws_on_bad_input():
#
#     with pytest.raises(ValueError):
#         net.vision_utilities.get_ordered_card_contour([])
#
#     with pytest.raises(ValueError):
#         net.vision_utilities.get_ordered_card_contour([1, 2, 3, 4, 5])
#
#     try:
#         net.vision_utilities.get_ordered_card_contour([1, 2, 3, 4])
#     except ValueError:
#         pytest.fail("Function failed, though shouldn't.")


# def test_get_ordered_card_contour_ordered_input():
#
#     contour = np.array(
#         [
#             [0, 0],
#             [0, 10],
#             [10, 10],
#             [10, 0]
#         ]
#     )
#
#     ordered_contour = net.vision_utilities.get_ordered_card_contour(contour)
#     assert np.all(contour == ordered_contour)
#
#
# def test_get_ordered_card_contour_simple_input():
#
#     contour = np.array(
#         [
#             [10, 10],
#             [10, 0],
#             [0, 0],
#             [0, 10]
#         ]
#     )
#
#     correct_contour = np.array(
#         [
#             [0, 0],
#             [0, 10],
#             [10, 10],
#             [10, 0]
#         ]
#     )
#
#     ordered_contour = net.vision_utilities.get_ordered_card_contour(contour)
#     assert np.all(correct_contour == ordered_contour)
