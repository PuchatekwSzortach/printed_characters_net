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


def test_is_point_above_region_midpoint_2():

    contour = np.array(
        [
            [2, 7],
            [8, 11],
            [10, 18],
            [-3, 16]
        ])

    assert True == net.vision_utilities.is_point_above_region_midpoint(contour, [2, 7])
    assert False == net.vision_utilities.is_point_above_region_midpoint(contour, [-3, 16])


def test_is_point_to_the_left_of_region_midpoint_1():

    contour = np.array(
        [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10]
        ])

    assert True == net.vision_utilities.is_point_to_the_left_of_region_midpoint(contour, [0, 0])
    assert False == net.vision_utilities.is_point_above_region_midpoint(contour, [10, 10])


def test_is_point_to_the_left_of_region_midpoint_2():

    contour = np.array(
        [
            [2, 7],
            [8, 11],
            [10, 18],
            [-3, 16]
        ])

    assert True == net.vision_utilities.is_point_to_the_left_of_region_midpoint(contour, [2, 7])
    assert False == net.vision_utilities.is_point_to_the_left_of_region_midpoint(contour, [8, 11])


def test_get_ordered_card_contour_throws_on_bad_input():

    with pytest.raises(ValueError):
        net.vision_utilities.get_ordered_card_contour(np.zeros(shape=(5,2)))

    with pytest.raises(ValueError):
        net.vision_utilities.get_ordered_card_contour(np.zeros(shape=(3,2)))


def test_get_ordered_card_contour_ordered_input():

    contour = np.array(
        [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
        ]
    )

    ordered_contour = net.vision_utilities.get_ordered_card_contour(contour)
    assert np.all(contour == ordered_contour)


def test_get_ordered_card_contour_simple_input():

    contour = np.array(
        [
            [10, 10],
            [10, 0],
            [0, 0],
            [0, 10]
        ]
    )

    correct_contour = np.array(
        [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10]
        ]
    )

    ordered_contour = net.vision_utilities.get_ordered_card_contour(contour)
    assert np.all(correct_contour == ordered_contour)


def test_get_ordered_card_contour_simple_complex_input():

    contour = np.array(
        [
            [7, 20],
            [-3, 16],
            [3, 8],
            [10, 14]
        ]
    )

    correct_contour = np.array(
        [
            [3, 8],
            [10, 14],
            [7, 20],
            [-3, 16],
        ]
    )

    ordered_contour = net.vision_utilities.get_ordered_card_contour(contour)
    assert np.all(correct_contour == ordered_contour)
