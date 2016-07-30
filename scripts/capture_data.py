"""
This program helps you capture data for training easily.
It asks you what character you intend to show it and then captures likely character locations
and saves in output directory at regular intervals.
Please note that module looking for likely character locations isn't perfect and sometimes makes mistakes,
so you would be best served manually confirming captured data is correct afterwards.
"""

import os
import cv2

import net.vision


def main():

    result = input("What character do you want to capture data for: ")[0]
    print("Ok, we will be capturing data for " + result)

    base_path = os.path.join("../../data/characters/data/", result)
    os.makedirs(base_path, exist_ok=True)

    video_capture = cv2.VideoCapture(0)

    while True:

        _, frame = video_capture.read()

        card_candidates = net.vision.CardCandidatesExtractor().get_card_candidates(frame)

        for candidate in card_candidates:

            cv2.drawContours(image=frame, contours=[candidate.coordinates],
                             contourIdx=0, color=(0, 255, 0), thickness=4)

        cv2.imshow("image", frame)

        key = cv2.waitKey(30)

        # If spacebar was pressed
        if key == 32:
            break


if __name__ == "__main__":
    main()