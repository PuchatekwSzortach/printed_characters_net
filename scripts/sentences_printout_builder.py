"""
A small utility for preparing pdfs with sentences
made out of card templates
"""
import glob
import os
import reportlab.platypus
import reportlab.lib.styles
import reportlab.lib.units
import reportlab.pdfgen.canvas
import reportlab.lib.pagesizes
import cv2
import net.printing
import shutil
import re


def get_characters_images_mapping(paths):

    mapping = {}

    for path in paths:

        character = os.path.basename(path)[0]
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

        mapping[character] = net.printing.get_image_with_border(image, 2)

    return mapping


def create_page(canvas, sentence, characters_paths_dictionary):
    """
    Create one page of text from text_iterator
    """

    height, width = reportlab.lib.pagesizes.A4
    image_size = int(width * 0.08)
    margin = int(width * 0.03)

    padded_image_size = image_size + (2 * margin)

    words = re.findall(r'\w+', sentence)

    for row, word in enumerate(words):

        y = height - padded_image_size - (row * padded_image_size)

        for column, character in enumerate(word):

            x = margin + (padded_image_size * column)

            canvas.drawImage(
                characters_paths_dictionary[character], x=x, y=y, height=image_size, width=image_size)

def main():

    output_path = "../../data/characters/sentence_printout.pdf"
    paths = glob.glob("../../data/characters/templates/*.jpg")

    temporary_dir = "/tmp/templates/"

    # Make directory for temporary images if it doesn't already exist
    os.makedirs(temporary_dir, exist_ok=True)

    # Write bordered images to temporary dir, get characters paths dictionary
    characters_path_dictionary = {}

    for path in paths:

        temporary_path = os.path.join(temporary_dir, os.path.basename(path + ".jpg"))
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        cv2.imwrite(temporary_path, net.printing.get_image_with_border(image, 2))

        characters_path_dictionary[os.path.basename(path)[0]] = temporary_path

    canvas = reportlab.pdfgen.canvas.Canvas(
            output_path, bottomup=1,
            pagesize=reportlab.lib.pagesizes.landscape(reportlab.lib.pagesizes.A4))

    create_page(canvas, "HIYANE SAN, ONISHI SAN", characters_path_dictionary)
    canvas.showPage()

    create_page(canvas, "HOW ABOUT", characters_path_dictionary)
    canvas.showPage()

    create_page(canvas, "WE MAKE ENGLISH KOTOBANBAN", characters_path_dictionary)
    canvas.showPage()

    canvas.save()

    # Clean up temporary images
    shutil.rmtree(temporary_dir)

if __name__ == "__main__":

    main()
