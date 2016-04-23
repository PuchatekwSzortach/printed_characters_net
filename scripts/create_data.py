"""
This script creates artificial training and test data from templates.
Each template will be randomly perturbed by a random selection of transformations.
Transformations used include shifts, rotations, white noise, foreground and background
intensity changes and perspective projections.
"""

import cv2

import net.transformations


def main():

    image = cv2.cvtColor(cv2.imread("../../data/characters/templates/a.jpg"), cv2.COLOR_RGB2GRAY)
    cv2.imshow("a", image)

    for index in range(10):

        shifted = net.transformations.add_noise(image)
        cv2.imshow(str(index), shifted)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()
