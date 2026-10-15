# Annotated was added to typing in Python 3.9; the Airflow image runs
# 3.8, so the backport is required for the DAG to import these steps.
import os
from typing import List

import pandas as pd
from typing_extensions import Annotated
from zenml import step

# Define the mapping logic here.
# We keep raw data in 3 folders, but train on 2 classes initially.
CLASS_MAPPING = {
    "fertile": "fertile",
    "infertile": "defect",
    "dead": "defect"
}

# The Roboflow export uses YOLO detection layout: images/ and labels/ side by
# side, with the class index as the first token of each label file. The class
# names come from the dataset's data.yaml.
YOLO_CLASS_NAMES = ["fertile", "infertile"]


def _ingest_yolo_layout(data_dir: str) -> pd.DataFrame:
    """Read a YOLO images/ + labels/ split into an image_path/label frame.

    The class of an image is taken from the first token of its label file. An
    image with no label file is skipped rather than guessed at, because an
    unlabelled image would silently become a wrong training example.
    """
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    if not os.path.isdir(images_dir):
        return pd.DataFrame(columns=["image_path", "label"])

    rows = []
    missing_labels = 0
    for filename in sorted(os.listdir(images_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        stem = os.path.splitext(filename)[0]
        label_file = os.path.join(labels_dir, f"{stem}.txt")
        if not os.path.exists(label_file):
            missing_labels += 1
            continue

        with open(label_file) as fh:
            first_line = fh.readline().strip()
        if not first_line:
            missing_labels += 1
            continue

        class_index = int(first_line.split()[0])
        if class_index >= len(YOLO_CLASS_NAMES):
            raise ValueError(
                f"{label_file} references class index {class_index}, but only "
                f"{len(YOLO_CLASS_NAMES)} classes are declared"
            )

        rows.append(
            {
                "image_path": os.path.abspath(os.path.join(images_dir, filename)),
                "label": YOLO_CLASS_NAMES[class_index],
            }
        )

    if missing_labels:
        print(f"   Skipped {missing_labels} image(s) with no usable label file")

    return pd.DataFrame(rows)


@step
def ingest_data(data_dir: str = "data/raw") -> Annotated[pd.DataFrame, "dataset"]:
    """
    Ingests data from the raw data directory and creates a DataFrame.

    Args:
        data_dir: Path to the raw data folder containing subfolders
                  (fertile, infertile, dead).

    Returns:
        pd.DataFrame: A dataframe containing 'image_path' and 'label'.
    """
    try:
        images: List[str] = []
        labels: List[str] = []

        print(f"🔍 Scanning data directory: {data_dir}")

        # Prefer the YOLO images/+labels/ layout when present: that is what the
        # Roboflow export actually ships, and the class-subdirectory layout below
        # would find nothing and raise "No images found".
        yolo_df = _ingest_yolo_layout(data_dir)
        if not yolo_df.empty:
            print(f"   Detected YOLO layout: {len(yolo_df)} labelled images")
            print(f"   Class Distribution: {yolo_df['label'].value_counts().to_dict()}")
            return yolo_df

        # We look for the folder names present in our mapping keys
        for folder_name, target_label in CLASS_MAPPING.items():
            folder_path = os.path.join(data_dir, folder_name)

            if not os.path.exists(folder_path):
                print(f"Warning: Folder '{folder_name}' not found. Skipping.")
                continue

            # List all valid images
            files = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            print(f"   Found {len(files)} images in '{folder_name}' -> Mapped to '{target_label}'")

            for filename in files:
                # Store absolute path to be safe
                full_path = os.path.abspath(os.path.join(folder_path, filename))
                images.append(full_path)
                labels.append(target_label)

        # Create DataFrame
        df = pd.DataFrame({"image_path": images, "label": labels})

        # Basic Validation
        if df.empty:
            raise ValueError("No images found! Check data/raw structure.")

        print(f"Ingestion Complete. Total images: {len(df)}")
        print(f"   Class Distribution: {df['label'].value_counts().to_dict()}")

        return df

    except Exception as e:
        print(f"Error in data ingestion: {e}")
        raise e
