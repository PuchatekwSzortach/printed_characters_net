"""
This is the "main" script of the entire program.
It reads in a net for recognizing templates and uses them to detect
templates in real time in camera images.
"""

import cv2

import net.vision
import net.network
import net.characters
import net.utilities
import numpy as np
import PIL.ImageFont
import PIL.Image
import PIL.ImageDraw


class CharactersDrawer:
    """
    A simple class for drawing characters on images
    """

    def __init__(self):
        pass

    def draw_character(self, image, character, character_outline):

        pil_image = PIL.Image.fromarray(image)

        font_size = int(0.5 * np.linalg.norm(character_outline[0] - character_outline[2]))
        font = PIL.ImageFont.truetype("/Library/Fonts/Osaka.ttf", size=font_size)

        draw = PIL.ImageDraw.ImageDraw(pil_image)
        draw.text(character_outline[2] - (0.5 * font_size), text=character, font=font, fill=(255, 0, 0))

        return np.array(pil_image)


def main():

    video_capture = cv2.VideoCapture(0)

    network = net.network.Net.from_file("./results/characters_net.json")
    base_data_path = "../../data/characters/data/"
    characters = net.characters.get_characters_list(base_data_path)
    encoder = net.utilities.Encoder(characters)

    characters_drawer = CharactersDrawer()

    while True:

        try:

            _, frame = video_capture.read()

            card_candidates = net.vision.CardCandidatesExtractor().get_card_candidates(frame)

            for candidate in card_candidates:

                transformed_image = net.characters.transform_image(candidate.image)
                prediction = network.feedforward(transformed_image)

                cv2.drawContours(image=frame, contours=[candidate.coordinates],
                                 contourIdx=0, color=(0, 255, 0), thickness=4)

                if np.max(prediction) > 0.5:

                    character = encoder.decode(prediction)

                    frame = characters_drawer.draw_character(frame, character, candidate.coordinates)

                    cv2.drawContours(image=frame, contours=[candidate.coordinates],
                                     contourIdx=0, color=(255, 0, 0), thickness=4)

            cv2.imshow("image", frame)

            key = cv2.waitKey(30)

            # If spacebar was pressed
            if key == 32:
                break

        except Exception as ex:

            print("An exception was caught")
            print(type(ex))
            print(ex)


if __name__ == "__main__":
    main()
