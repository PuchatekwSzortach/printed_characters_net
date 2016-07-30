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


def main():

    video_capture = cv2.VideoCapture(0)

    network = net.network.Net.from_file("./results/characters_net.json")
    base_data_path = "../../data/characters/data/"
    characters = net.characters.get_characters_list(base_data_path)
    encoder = net.utilities.Encoder(characters)

    while True:

        _, frame = video_capture.read()

        card_candidates = net.vision.CardCandidatesExtractor().get_card_candidates(frame)

        for candidate in card_candidates:

            transformed_image = net.characters.transform_image(candidate.image)
            prediction = network.feedforward(transformed_image)

            cv2.drawContours(image=frame, contours=[candidate.coordinates],
                             contourIdx=0, color=(0, 255, 0), thickness=4)

            if np.max(prediction) > 0.5:
                print(encoder.decode(prediction))

                cv2.drawContours(image=frame, contours=[candidate.coordinates],
                                 contourIdx=0, color=(255, 0, 0), thickness=4)



        cv2.imshow("image", frame)

        key = cv2.waitKey(30)

        # If spacebar was pressed
        if key == 32:
            break

if __name__ == "__main__":
    main()
