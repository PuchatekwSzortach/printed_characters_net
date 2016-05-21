# -*- coding: utf-8 -*-

"""
This script creates artificial training and test data from templates.
Each template will be randomly perturbed by a random selection of transformations.
Transformations used include shifts, rotations, white noise, foreground and background
intensity changes and perspective projections.
"""

import random
import numpy as np
import cv2
import glob
import os.path

import net.transformations


def get_file_name(template_name, index_number):

    stem = "../../data/characters/data/" + template_name + "/"
    file_name = "{:04}".format(index_number)

    return stem + file_name + ".jpg"


def main():

    tranformations = {
        "shift": net.transformations.shift_image,
        "rotation": net.transformations.rotate_image,
        "intensity_change": net.transformations.change_intensity,
        "noise": net.transformations.add_noise,
        "perspective_transformation": net.transformations.apply_perspective_transformation,
    }

    templates = glob.glob("../../data/characters/templates/*.jpg")

    for template in templates:

        template_name = os.path.split(template)[1].split(".")[0]

        directory = "../../data/characters/data/" + template_name

        if not os.path.exists(directory):
            os.makedirs(directory)

        image = cv2.cvtColor(cv2.imread(template), cv2.COLOR_RGB2GRAY)

        for index in range(100):

            # Get number of transformations, then get actual transformations in random order
            transformations_count = np.random.randint(1, len(tranformations.keys()) + 1)
            applied_transformations = random.sample(tranformations.keys(), transformations_count)

            transformed_image = image.copy()

            for key in applied_transformations:

                transformed_image = tranformations[key](transformed_image)
                cv2.imwrite(get_file_name(template_name, index), transformed_image)


if __name__ == "__main__":
    main()
