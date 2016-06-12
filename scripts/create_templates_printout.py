"""
A script for creating a pdf with all templates.
"""

import reportlab.platypus
import reportlab.lib.styles
import reportlab.lib.units
import reportlab.pdfgen.canvas
import reportlab.lib.pagesizes

import glob
import cv2
import numpy as np

import PIL.Image
import os.path
import shutil

def get_image_with_border(image, border_width):
    """
    Add a border around image
    :param image: image
    :param border_width: border width
    :return: image with black border around it
    """
    height, width = image.shape
    bordered_shape = ((2 * border_width) + height, (2 * border_width) + width)
    bordered_image = np.zeros(bordered_shape)

    bordered_image[border_width: height + border_width, border_width: width + border_width] = image
    return bordered_image

def create_page(canvas, canvas_size, paths_iterator, image_size, margin):

    padded_image_size = image_size + (2 * margin)
    width, height = canvas_size

    columns = int(width / padded_image_size)
    rows = int(height / padded_image_size)

    y = margin

    for _ in range(rows):

        x = margin

        for _ in range(columns):

            canvas.drawImage(
                next(paths_iterator), x=x, y=y, height=image_size, width=image_size)

            x += padded_image_size

        y += padded_image_size


def main():

    output_path = "../../data/characters/templates_printout.pdf"

    canvas = reportlab.pdfgen.canvas.Canvas(
        output_path, bottomup=1, pagesize=reportlab.lib.pagesizes.A4)

    canvas_size = reportlab.lib.pagesizes.A4
    width, height = canvas_size

    image_size = int(width * 0.3)
    margin = int(width * 0.015)

    paths = glob.glob("../../data/characters/templates/*.jpg")

    templates = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY) for path in paths]
    bordered_templates = [get_image_with_border(template, 2) for template in templates]

    temporary_dir = "/tmp/templates/"

    try:
        os.makedirs(temporary_dir)
    except os.error:
        pass

    for path, template in zip(paths, bordered_templates):
        temporary_path = temporary_dir + os.path.basename(path)
        cv2.imwrite(temporary_path, template)

    paths_iterator = iter(glob.glob(temporary_dir + "/*.jpg"))

    # Create pages until we run out of template paths
    while True:

        try:

            create_page(canvas, canvas_size, paths_iterator, image_size, margin)
            canvas.showPage()

        except StopIteration:

            break

    canvas.save()

    shutil.rmtree(temporary_dir)

if __name__ == "__main__":
    main()