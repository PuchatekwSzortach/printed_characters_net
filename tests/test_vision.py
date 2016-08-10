"""
Tests for vision_utilities module
"""
import numpy as np
import pytest

import net.vision

def test_get_ordered_card_contour_throws_on_bad_input():

    with pytest.raises(ValueError):
        net.vision.get_ordered_card_contour(np.zeros(shape=(5,2)))

    with pytest.raises(ValueError):
        net.vision.get_ordered_card_contour(np.zeros(shape=(3,2)))


def test_get_ordered_card_contour_ordered_input():

    contour = np.array(
        [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
        ]
    )

    ordered_contour = net.vision.get_ordered_card_contour(contour)
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

    ordered_contour = net.vision.get_ordered_card_contour(contour)
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

    ordered_contour = net.vision.get_ordered_card_contour(contour)
    assert np.all(correct_contour == ordered_contour)


def test_get_minimum_inner_angle_square_input():

    contour = np.array(
        [
            [0, 0],
            [0, 10],
            [10, 10],
            [10, 0]
        ]
    )

    assert np.isclose(np.pi / 2, net.vision.get_minimum_inner_angle(contour))


def test_get_minimum_inner_angle_30deg_triangle():

    contour = np.array(
        [
            [0, 0],
            [5 * np.sqrt(3), 0],
            [5 * np.sqrt(3), 5],
        ]
    )

    assert np.isclose(np.pi / 6, net.vision.get_minimum_inner_angle(contour))


def test_get_maximum_inner_angle_square_input():

    contour = np.array(
        [
            [0, 0],
            [0, 10],
            [10, 10],
            [10, 0]
        ]
    )

    assert np.isclose(np.pi / 2, net.vision.get_minimum_inner_angle(contour))


def test_get_maximum_inner_angle_135deg_triangle():

    contour = np.array(
        [
            [0, 0],
            [10, 0],
            [20, 10]
        ]
    )

    assert np.isclose(135 * np.pi / 180, net.vision.get_maximum_inner_angle(contour))