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

    max_shift = int(max(image.shape) / 20)
    y_shift, x_shift = np.random.randint(-max_shift, max_shift, [2])

    transformation_matrix = np.eye(3)[:2, :]
    transformation_matrix[0, 2] = x_shift
    transformation_matrix[1, 2] = y_shift

    return cv2.warpAffine(image, transformation_matrix, dsize=image.shape)


def rotate_image(image):
    """
    Rotate image by a random amount proportional to image size.
    Rotation centre is a random point inside the original image.
    :param image:
    :return: rotated image. Note that image will have black artifacts around corners. This is left
    in output on purpose, as should create a more difficult data set.
    """

    # Get rotation amount in degrees
    max_angle = 3
    angle = np.random.randint(-max_angle, max_angle)

    # Get rotation centre
    y, x = np.random.randint(0, image.shape[0]), np.random.randint(0, image.shape[0])

    rotation_matrix = cv2.getRotationMatrix2D((x, y), angle, 1)
    return cv2.warpAffine(image, rotation_matrix, dsize=image.shape)


def change_intensity(image, background_change=None, foreground_change=None):
    """
    Change background and foreground intensity by a random value
    :param image: Image with intensity of background and foreground changed
    :param background_change: intensity change of background. Random if none
    :param foreground_change: intensity change of background. Random if none
    :return: Altered image
    """

    if background_change is None:
        background_change = np.random.randint(10, 50)

    if foreground_change is None:
        foreground_change = np.random.randint(10, 50)

    altered_image = image.copy()

    # We assume black is foreground (character) and white it background
    altered_image[altered_image < 10] += foreground_change
    altered_image[altered_image > 200] -= background_change

    return altered_image


def add_noise(image):
    """
    Adds uniform noise to image
    :param image:
    :return: image with uniform noise applied to it
    """

    noise_location = np.random.binomial(1, 0.1, image.shape)
    noise = (np.random.randint(0, 100, image.shape) * noise_location).astype(image.dtype)

    noisy_image = image.copy()
    noisy_image[noisy_image < 20] += noise[noisy_image < 20]
    noisy_image[noisy_image > 200] -= noise[noisy_image > 200]

    return noisy_image


def apply_perspective_transformation(image):
    """
    Apply a random perspective transformation to image
    :param image:
    :return: Image skewed by perspective transformation.
    Note that image will have black artifacts around corners. This is left
    in output on purpose, as should create a more difficult data set.
    """

    corners = np.zeros([4, 2], dtype=np.float32)

    corners[0, :] = [0, 0]
    corners[1, :] = [image.shape[1], 0]
    corners[2, :] = [image.shape[1], image.shape[0]]
    corners[3, :] = (0, image.shape[0])

    max_distortion = max(image.shape) / 40
    distorted_corners = corners + np.random.uniform(-max_distortion, max_distortion, corners.shape).astype(np.float32)

    transformation_matrix = cv2.getPerspectiveTransform(corners, distorted_corners)
    return cv2.warpPerspective(image, transformation_matrix, image.shape)


def blur_image(image, kernel=None):
    """
    Blur image using normalized box filter
    :param image: image to blur
    :param kernel: kernel specifying strength of blur
    :return: blurred image
    """

    if kernel is None:
        kernel = np.random.randint(1, 5, size=[2])

    return cv2.blur(image, tuple(kernel))


def trim_border(image):
    """
    Randomly trim off a few pixels from around the border
    :param image:
    :return: image with border removed. Resized to initial image size
    """

    border = np.random.randint(1, int(0.1 * np.max(image.shape)))
    new_image = image[border:image.shape[0] - border, border: image.shape[1] - border].copy()
    return cv2.resize(new_image, image.shape)
