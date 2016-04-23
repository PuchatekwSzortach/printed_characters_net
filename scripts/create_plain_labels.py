# -*- coding: utf-8 -*-

import os
import sys

import PIL.ImageFont
import cv2

# sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
import net.data


def main():

    font = PIL.ImageFont.truetype("/Library/Fonts/Osaka.ttf", size=100)

    characters = "ABCDE"

    card_maker = net.data.PlainCardMaker(font, (64, 64))

    for character in characters:

        image = card_maker.create_image(character)
        cv2.imwrite("../../data/characters/templates/" + character + ".jpg", image)

if __name__ == "__main__":

    main()
