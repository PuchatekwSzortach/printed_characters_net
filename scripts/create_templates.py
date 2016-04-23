# -*- coding: utf-8 -*-

"""
This file creates templates for characters we want to be able to detect.
Templates are simple, front facing images without any distortions, blemishes, etc.
"""

import os
import sys

import PIL.ImageFont
import cv2

# sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
import net.data


def main():

    font = PIL.ImageFont.truetype("/Library/Fonts/Osaka.ttf", size=100)

    characters = "ABCDEFGHIJKLMN"

    templates_maker = net.data.TemplatesMaker(font, (64, 64))

    for character in characters:

        image = templates_maker.create_template(character)
        cv2.imwrite("../../data/characters/templates/" + character + ".jpg", image)

if __name__ == "__main__":

    main()
