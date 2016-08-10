"""
Module with computer vision related code.
Detecting card candidates in images, handling image contours and like.
"""
import cv2
import numpy as np


class CardCandidate:
    """
    A very simple container for a card candidate
    """

    def __init__(self, coordinates, image):

        self.coordinates = coordinates
        self.image = image


class CardCandidatesExtractor:
    """
    Class for extracting card candidates from an input image
    """

    def get_card_candidates(self, image, reconstruction_size):

        image_contours = self._get_image_contours(image)
        card_contours = self._get_card_like_contours(image_contours, image.shape[0] * image.shape[1])
        outer_contours = self._get_outermost_contours(card_contours, image.shape)

        # OpenCV puts into contour an unnecessary dimension, so remove it
        squeezed_contours = [np.squeeze(contour) for contour in outer_contours]

        reconstruction_contour = np.array([
            [0, 0],
            [reconstruction_size[1], 0],
            [reconstruction_size[1], reconstruction_size[0]],
            [0, reconstruction_size[0]]
        ])

        # We need to make sure ordering within each contour is consistent
        ordered_contours = [get_ordered_card_contour(contour) for contour in squeezed_contours]

        card_candidates = []

        for contour in ordered_contours:

            reconstruction = get_card_reconstruction(
                image, contour, reconstruction_contour, reconstruction_size)

            card_candidates.append(CardCandidate(contour, reconstruction))

        return card_candidates

    def _get_image_contours(self, image):

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, cv2.THRESH_BINARY)
        thresholded = self._get_thresholded_image(grayscale)

        _, contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        return contours

    def _get_thresholded_image(self, grayscale_image):

        return cv2.adaptiveThreshold(
            src=grayscale_image, maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY, blockSize=9, C=-5)

    def _get_card_like_contours(self, contours, image_size):

        epsilon = image_size * 1e-5

        simplified_contours = [cv2.approxPolyDP(contour, epsilon=epsilon, closed=True) for contour in contours]

        card_like_contours = [
            contour for contour in simplified_contours
            if is_contour_card_like(contour, image_size)]

        return card_like_contours

    def _get_outermost_contours(self, contours, image_shape):

        image = np.zeros(shape=(image_shape[0], image_shape[1])).astype(np.uint8)
        cv2.drawContours(image, contours, contourIdx=-1, color=255)

        # Use OpenCV for heavy lifting
        _, outer_contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Run through sanity check, just to be extra careful, and return results
        return self._get_card_like_contours(outer_contours, image_shape[0] * image_shape[1])


def get_card_reconstruction(image, contour, reconstruction_contour, reconstruction_size):
    """
    Given an image, a contour inside it and a reconstruction_contour,
        map image contained inside contour to reconstruction contour.
        Both contours are assumed to have exactly 4 points.
    :param image: image to reconstruct from
    :param contour: contour around area to recontruct from image
    :param reconstruction_contour: contour of intended reconstruction
    :return: CardReconstruction object
    """

    # Do a sanity check
    if len(contour) != 4 or len(reconstruction_contour) != 4:
        message = "Both contour (len = {}) and reconstruction contour (len = {}) "
        "should have exactly 4 points.".format(len(contour), len(reconstruction_contour))
        raise ValueError(message)

    transformation_matrix = cv2.getPerspectiveTransform(
        contour.astype(np.float32), reconstruction_contour.astype(np.float32))

    shape = np.max(reconstruction_contour, axis=0).astype(np.int32)

    reconstruction = cv2.warpPerspective(image, transformation_matrix, tuple(shape))

    resized_reconstruction = cv2.resize(reconstruction, reconstruction_size)
    grayscale_reconstruction = cv2.cvtColor(resized_reconstruction, cv2.COLOR_RGB2GRAY)
    return grayscale_reconstruction


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

    ordered_contour = np.zeros_like(contour)

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


def is_contour_card_like(contour, image_size):
    """
    Given a contour, judge whether it is like to represent a card.
    The criteria are that it is roughly rectangular and has a sensible area w.r.t.
    total image size
    :param contour: contour to judge
    :param image_size: number of pixels in image
    :return: Bool
    """

    max_area = 0.3 * image_size
    min_area = 0.001 * image_size

    # Only contours with 4 points can be card candidates
    if len(contour) != 4:
        return False

    contour_area = cv2.contourArea(contour)

    # Contours must be within acceptable size
    if contour_area < min_area or max_area < contour_area:
        return False

    squeezed_contour = np.squeeze(contour)
    min_angle = get_minimum_inner_angle(squeezed_contour)
    max_angle = get_maximum_inner_angle(squeezed_contour)

    # And have inner angles somewhat close to right angles
    if min_angle < np.pi / 6 or np.pi / 1.5 < max_angle:
        return False

    # All tests passed, so judge contour as card like
    return True


def get_contours_inner_angles(contour):
    """
    Given a closed contour, return a list of its inner angles in radians
    :param contour: a 2D numpy array
    :return: 1D numpy array of angles in radions
    """

    # Unroll all points triplets to vectors
    a = contour
    b = np.tile(contour, reps=(2, 1))[1:1 + len(contour)]
    c = np.tile(contour, reps=(2, 1))[2:2 + len(contour)]

    a_side = np.linalg.norm(b - a, axis=1)
    b_side = np.linalg.norm(c - b, axis=1)
    c_side = np.linalg.norm(c - a, axis=1)

    cosines = (a_side**2 + b_side**2 - c_side**2) / (2 * a_side * b_side)
    return np.arccos(cosines)


def get_minimum_inner_angle(contour):
    """
    Given a list of 2D points representing a closed contour,
    return smallest inner angle
    :param contour: a 2D numpy array
    :return: angle in radians
    """

    return np.min(get_contours_inner_angles(contour))


def get_maximum_inner_angle(contour):
    """
    Given a list of 2D points representing a closed contour,
    return largest inner angle
    :param contour: a 2D numpy array
    :return: angle in radians
    """

    return np.max(get_contours_inner_angles(contour))
