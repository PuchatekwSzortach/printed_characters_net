"""
This program helps you capture data for training easily.
It asks you what character you intend to show it and then captures likely character locations
and saves in output directory at regular intervals.
Please note that module looking for likely character locations isn't perfect and sometimes makes mistakes,
so you would be best served manually confirming captured data is correct afterwards.
"""

import os
import cv2
import glob
import re
import configobj

import net.vision


def save_file(dir_path, index, image):

    file_name = "{:05}.jpg".format(index)
    output_path = os.path.join(dir_path, file_name)
    cv2.imwrite(output_path, image)


def get_initial_file_counter_value(base_path):

    existing_files = glob.glob(os.path.join(base_path, "*.jpg"))

    if len(existing_files) == 0:
        return 1
    else:
        last_file = os.path.basename(existing_files[-1])
        last_file_index = int(re.findall(r'\d+', last_file)[0])
        return last_file_index + 1


def main():

    result = input("What character do you want to capture data for: ")[0]
    print("Ok, we will be capturing data for " + result)

    base_path = os.path.join("../../data/characters/data/", result)
    os.makedirs(base_path, exist_ok=True)

    file_counter = get_initial_file_counter_value(base_path)
    previous_files_count = file_counter - 1

    if file_counter > 1:
        print("We will be starting from file {}".format(file_counter))

    frame_counter = 0

    video_capture = cv2.VideoCapture(0)

    config = configobj.ConfigObj('configuration.ini')
    reconstruction_size = tuple([int(value) for value in config['image_size']])

    while True:

        _, frame = video_capture.read()

        card_candidates = net.vision.CardCandidatesExtractor().get_card_candidates(frame, reconstruction_size)

        for candidate in card_candidates:

            cv2.drawContours(image=frame, contours=[candidate.coordinates],
                             contourIdx=0, color=(0, 255, 0), thickness=4)

        # Only save data roughly three times per second
        if frame_counter == 10:

            for candidate in card_candidates:

                save_file(base_path, file_counter, candidate.image)

                print("Captured {} images".format(file_counter - previous_files_count), end='\r')
                file_counter += 1

            frame_counter = 0

        cv2.imshow("image", frame)

        key = cv2.waitKey(30)

        # If spacebar was pressed
        if key == 32:
            break

        frame_counter += 1


if __name__ == "__main__":
    main()