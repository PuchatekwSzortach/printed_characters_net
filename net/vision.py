"""
Module with computer vision related code.
Detecting card candidates in images, handling image contours and like.
"""
import cv2
import numpy as np


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
        _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours


