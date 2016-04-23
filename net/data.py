"""
Module with code related to creating data.
"""

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

import numpy as np
import cv2


class TemplatesMaker:
    """
    Class for creating plain images of characters.
    Plain image is defined as a simple black character on white background. No noise, rotation, etc included.
    """

    def __init__(self, font, size):
        """
        :param font: PIL.ImageFont's truefont object to be used to create character image
        :param size: a (width, height) tuple representing size of images to be produced
        """

        self.font = font
        self.size = size

    def create_template(self, character):
        """
        :param character: character to be shown in the image
        :return: numpy matrix representing the character
        """

        image = self._get_character_image(character)

        # Cut out portion that contains characters
        upper_left, lower_right = self._get_character_bounding_box(image)

        # Image cropped to contain only character
        character_image = image[upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]]

        larger_dimension = max(character_image.shape)

        padding = 20
        full_image_size = larger_dimension + 2 * padding, larger_dimension + 2 * padding
        full_image = 255 * np.ones(full_image_size)

        image_center = int(full_image_size[0] / 2), int(full_image_size[1] / 2)

        y_start = int(image_center[0] - (character_image.shape[0] / 2))
        x_start = int(image_center[1] - (character_image.shape[1] / 2))

        # Paste character image onto full image
        full_image[
            y_start:y_start + character_image.shape[0],
            x_start:x_start + character_image.shape[1]] = character_image

        return cv2.resize(full_image, self.size)

    def _get_character_image(self, character):

        x_size, y_size = self.font.getsize(character)

        image = PIL.Image.new("RGB", (2 * x_size, 2 * y_size), "white")

        draw = PIL.ImageDraw.ImageDraw(image)
        draw.text((x_size / 2, y_size / 2), text=character, font=self.font, fill=(0, 0, 0))

        return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    def _get_character_bounding_box(self, image):

        # Image has white background (255 value) and black letter (0 value).
        # So to find bounding box find locations of columns and rows that have zero elements
        zero_rows = np.any(image == 0, axis=1)
        zero_rows_indices = np.nonzero(zero_rows)[0]

        zero_columns = np.any(image == 0, axis=0)
        zero_columns_indices = np.nonzero(zero_columns)[0]

        return (zero_rows_indices[0], zero_columns_indices[0]), \
               (zero_rows_indices[-1], zero_columns_indices[-1])
