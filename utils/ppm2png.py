"""Small helper script to convert images from ppm to png format.

Recursively searches the specified directory for ppm files.

Example:
    >>> python -m ppm2png -d ./GTSRB_Final_Training_Images/GTSRB/Final_Training/Images -c 1

Note:
    Uses openCV which can be installed for example via pip:
    >>> pip install opencv-python
    You can also use Pillow instead.

"""
import argparse
from pathlib import Path

import cv2

CWD = Path.cwd()


def convert_ppm_to_png_images(directory_path: Path, delete_ppm_files: bool) -> None:
    for filepath in directory_path.glob("**/*.ppm"):
        img = cv2.imread(str(filepath))
        cv2.imwrite(str(filepath).replace(".ppm", ".png"), img)
        if delete_ppm_files:
            filepath.unlink()
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        help="Relative path to the root directory of images that should be converted from ppm to png format.",
    )
    parser.add_argument(
        "-c",
        "--clean",
        help="Indicating whether ppm files should be deleted after conversion.",
        type=int,
        choices=[0, 1],
    )
    args = parser.parse_args()

    img_root_directory_path = CWD / args.directory
    delete_ppm_files = args.clean == 1

    convert_ppm_to_png_images(img_root_directory_path, delete_ppm_files)