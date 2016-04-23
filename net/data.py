"""
Module with code related to creating data.
"""

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

import numpy as np
import cv2


class PlainCardMaker:
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

    def create_image(self, character):
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
        upper_left, lower_right = self._get_character_boundig_box(image)
        return image[upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]]

    def _get_character_boundig_box(self, image):

        contours = cv2.findContours(image.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[1]

        x_coordinates = [point[0][1] for contour in contours for point in contour]
        y_coordinates = [point[0][0] for contour in contours for point in contour]


        print(min(y_coordinates))
        print(min(x_coordinates))

        print(max(y_coordinates))
        print(max(x_coordinates))

        return (20, 10), (80, 50)
