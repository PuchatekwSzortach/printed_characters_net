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

import os.path
import shutil
import net.printing



def create_page(canvas, canvas_size, paths_iterator, image_size, margin):
    """
    Create one page of templates. This function will throw a StopIteration exception
     when paths_iterator runs out.
    """

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

    # Size of image and margin between images on a pdf page
    image_size = int(width * 0.2)
    margin = int(width * 0.06)

    paths = glob.glob("../../data/characters/templates/*.jpg")

    templates = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY) for path in paths]
    temporary_dir = "/tmp/templates/"

    # Make directory for temporary images if it doesn't already exist
    os.makedirs(temporary_dir, exist_ok=True)

    # Write bordered images to temporary dir
    for path, template in zip(paths, templates):
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

    # Clean up temporary images
    shutil.rmtree(temporary_dir)

if __name__ == "__main__":
    main()