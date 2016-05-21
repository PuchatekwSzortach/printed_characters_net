"""
Module for training net to recognize printed characters
"""

import os

def get_characters_list(data_path):
    """
    Give a path which contains directories with data for each character,
    return a list with characters we have data for
    :param data_path: path to data directory. It is assumed to contain
    directories names after characters, with each directory containing images of that character
    :return: a list of characters for which we have images
    """

    # First result contains parent directory itself, so need to discard it.
    # Also full paths are given and we want only basenames
    return [os.path.basename(result[0]) for result in os.walk(data_path)][1:]

def main():

    characters = get_characters_list("../../data/characters/data")
    

if __name__ == "__main__":
    main()