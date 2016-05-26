"""
A script for creating a pdf with all templates.
"""

import reportlab.platypus
import reportlab.lib.styles
import reportlab.lib.units
import reportlab.pdfgen.canvas
import reportlab.lib.pagesizes

import glob

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

    templates_paths = glob.glob("../../data/characters/templates/*.jpg")
    paths_iterator = iter(templates_paths)

    # Create pages until we run out of template paths
    while True:

        try:

            create_page(canvas, canvas_size, paths_iterator, image_size, margin)
            canvas.showPage()

        except StopIteration:
            break


    canvas.save()


if __name__ == "__main__":
    main()