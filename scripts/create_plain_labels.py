# -*- coding: utf-8 -*-

import os
import sys

import PIL.ImageFont
import cv2

# sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
import net.data


def main():

    font = PIL.ImageFont.truetype("/Library/Fonts/Osaka.ttf", size=50)

    characters = "ABCD舞弥藍鸞"

    card_maker = net.data.PlainCardMaker(font, (64, 64))

    for character in characters:

        image = card_maker.create_image(character)
        cv2.imshow(character, image)

    cv2.waitKey(0)

if __name__ == "__main__":

    main()
