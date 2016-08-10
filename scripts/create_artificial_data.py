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
import multiprocessing

import tqdm

import net.transformations


def get_file_name(template_name, index_number):

    stem = "../../data/characters/data/" + template_name + "/"
    file_name = "{:04}".format(index_number)

    return stem + file_name + ".jpg"


def create_template_data(template_path, base_path, transformations, images_count):

    template_name = os.path.split(template_path)[1].split(".")[0]

    directory = os.path.join(base_path, template_name)
    os.makedirs(directory, exist_ok=True)

    image = cv2.cvtColor(cv2.imread(template_path), cv2.COLOR_RGB2GRAY)

    # First apply an intensity transformation to template
    image = net.transformations.change_intensity(image, 75, 10)

    # Caputer images are often somewhat blurred, so apply a light blurring
    image = net.transformations.blur_image(image, (3, 3))

    # A series of random integers representing number of transformations to
    # be applied for each image
    transformations_counts = np.random.randint(1, len(transformations.keys()) + 1, size=images_count)

    for image_index in range(images_count):

        # Get number of transformations, then get actual transformations in random order
        applied_transformations = random.sample(
            transformations.keys(), transformations_counts[image_index])

        transformed_image = image.copy()

        for key in applied_transformations:

            transformed_image = transformations[key](transformed_image)
            cv2.imwrite(get_file_name(template_name, image_index), transformed_image)


def main():

    transformations = {
        "shift": net.transformations.shift_image,
        "rotation": net.transformations.rotate_image,
        "intensity_change": net.transformations.change_intensity,
        "noise": net.transformations.add_noise,
        "perspective_transformation": net.transformations.apply_perspective_transformation,
        "blur": net.transformations.blur_image,
        "trim_border": net.transformations.trim_border
    }

    templates_paths = glob.glob("../../data/characters/templates/*.jpg")
    base_path = "../../data/characters/data/"
    images_count = 500

    with multiprocessing.Pool() as pool:

        # Enqueue jobs using processing pool
        results = [pool.apply_async(create_template_data, (path, base_path, transformations, images_count))
                   for path in templates_paths]

        # Wrap waiting on results into a nifty progress bar
        for result in tqdm.tqdm(results):
            result.get()


if __name__ == "__main__":
    main()
