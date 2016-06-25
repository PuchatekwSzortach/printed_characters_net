"""
Module with computer vision related code.
Detecting card candidates in images, handling image contours and like.
"""
import cv2
import numpy as np

import net.vision_utilities
import net.constants


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

    def get_card_candidates(self, image):

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, cv2.THRESH_BINARY)
        thresholded = self._get_thresholded_image(grayscale)

        _, contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        card_contours = self._get_card_like_contours(contours, image.shape[0] * image.shape[1])

        outer_contours = self._get_outermost_contours(card_contours, thresholded.shape)

        # OpenCV puts into contour an unnecessary dimension, so remove it
        squeezed_contours = [np.squeeze(contour) for contour in outer_contours]

        # We need to make sure ordering within each contour is consistent
        ordered_contours = [net.vision_utilities.get_ordered_card_contour(contour)
                            for contour in squeezed_contours]

        card_candidates = []

        for contour in ordered_contours:

            reconstruction = get_card_reconstruction(
                image, contour, net.constants.straigt_card_coordinates)

            card_candidates.append(CardCandidate(ordered_contours, reconstruction))

        return card_candidates


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
            if self._is_contour_card_like(contour, image_size)]

        return card_like_contours

    def _is_contour_card_like(self, contour, image_size):

        max_area = 0.3 * image_size
        min_area = 0.001 * image_size

        contour_area = cv2.contourArea(contour)

        # Return contours that are within acceptable size
        if contour_area < min_area or max_area < contour_area:
            return False

        # And made of only four points - since our frames should be represented by
        # four points
        return len(contour) == 4

    def _get_outermost_contours(self, contours, image_shape):

        image = np.zeros(shape=image_shape).astype(np.uint8)
        cv2.drawContours(image, contours, contourIdx=-1, color=255)

        # Use OpenCV for heavy lifting
        _, outer_contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Run through sanity check, just to be extra careful, and return results
        return self._get_card_like_contours(outer_contours, image_shape[0] * image_shape[1])


def get_card_reconstruction(image, contour, reconstruction_contour):
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
    return reconstruction






