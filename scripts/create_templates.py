# -*- coding: utf-8 -*-

"""
This file creates templates for characters we want to be able to detect.
Templates are simple, front facing images without any distortions, blemishes, etc.
"""

import PIL.ImageFont
import cv2
import net.data
import string
import os

def get_hiragana_set():

    return [chr(index) for index in range(12354, 12439)]

def get_katakana_set():

    return [chr(index) for index in range(12447, 12544)]

def get_digits_set():

    return [str(index) for index in range(0, 10)]

def get_kanji_set():

    with open("../../data/characters/kanji.txt") as file:
        content = file.read().split(" ")
        return content

def main():


    font = PIL.ImageFont.truetype("/Library/Fonts/Osaka.ttf", size=100)

    characters = list(string.ascii_uppercase) + get_hiragana_set() + \
                 get_katakana_set() + get_digits_set() + get_kanji_set()

    templates_maker = net.data.TemplatesMaker(font, (32, 32))

    base_path = "../../data/characters/templates/"
    os.makedirs(base_path)

    for character in characters:

        image = templates_maker.create_template(character)
        cv2.imwrite(base_path + character + ".jpg", image)

if __name__ == "__main__":

    main()
