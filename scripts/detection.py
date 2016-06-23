"""
This is the "main" script of the entire program.
It reads in a net for recognizing templates and uses them to detect
templates in real time in camera images.
"""

import cv2

import net.vision

def main():

    video_capture = cv2.VideoCapture(0)

    while True:

        _, frame = video_capture.read()

        net.vision.CardCandidatesExtractor().get_card_candidates(frame)
        cv2.imshow("image", frame)

        key = cv2.waitKey(30)

        # If spacebar was pressed
        if key == 32:
            break

if __name__ == "__main__":
    main()