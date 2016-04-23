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

        x_size, y_size = self.font.getsize(character)

        image = PIL.Image.new("RGB", (2 * x_size, 2 * y_size), "white")
        draw = PIL.ImageDraw.ImageDraw(image)

        draw.text((x_size / 2, y_size / 2), text=character, font=self.font, fill=(0, 0, 0))

        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

        # Cut out portion that contains characters
        upper_left, lower_right = self._get_character_bounding_box(image)

        # Image cropped to contain only character
        character_image = image[upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]]

        padding = 20
        full_image_size = character_image.shape[0] + 2 * padding, character_image.shape[1] + 2 * padding
        full_image = 255 * np.ones(full_image_size)

        # Paste character image onto full image
        full_image[
            padding:full_image_size[0] - padding,
            padding:full_image_size[1] - padding] = character_image

        return cv2.resize(full_image, self.size)

    def _get_character_bounding_box(self, image):

        # Image has white background (255 value) and black letter (0 value).
        # So to find bounding box find locations of columns and rows that have zero elements
        zero_rows = np.any(image == 0, axis=1)
        zero_rows_indices = np.nonzero(zero_rows)[0]

        zero_columns = np.any(image == 0, axis=0)
        zero_columns_indices = np.nonzero(zero_columns)[0]

        return (zero_rows_indices[0], zero_columns_indices[0]), \
               (zero_rows_indices[-1], zero_columns_indices[-1])
