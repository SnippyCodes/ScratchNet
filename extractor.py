import argparse
from pathlib import Path

import cv2
import numpy as np

categories = ["daisy", "dandelion", "rose", "sunflower", "tulip", "bellflower", "lotus"]
target_size = (150, 150)

def pixel_converter(dataset_dir):
    resized_pixels = []
    labelled = []

    for category in categories:
        folder = dataset_dir / category
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing category folder: {folder}\n"
                "Download/extract the raw flower images and pass their category root "
                "with --dataset-dir. Expected layout: <dataset-dir>/<category>/image.jpg"
            )
        count_in_category = 0

        for file_path in folder.iterdir():
            read_file = cv2.imread(str(file_path))

            if read_file is None:
                print(f"Skipping unreadable file: {file_path}")
                continue

            resized_image = cv2.resize(read_file, target_size)
            resized_pixels.append(resized_image)

            category_index = categories.index(category)
            labelled.append(category_index)

            count_in_category += 1

        print(f"{category}: {count_in_category} images processed")

    return resized_pixels, labelled


parser = argparse.ArgumentParser(description="Convert flower images into an NPZ dataset.")
parser.add_argument(
    "--dataset-dir",
    type=Path,
    default=Path(__file__).resolve().parent / "flowers" / "flowers",
    help="Folder containing one subfolder per flower category.",
)
args = parser.parse_args()

# Run the extraction
resized_pixels, labelled = pixel_converter(args.dataset_dir)

# Convert to numpy arrays
X = np.array(resized_pixels)
Y = np.array(labelled)


# Save both arrays together in the NPZ format used by the training notebook.
np.savez("flowers_data.npz", X=X, Y=Y)
print("Saved flowers_data.npz successfully.")