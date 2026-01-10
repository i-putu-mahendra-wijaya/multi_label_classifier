from typing import Tuple, List, Set, Dict

import os
import pathlib
from PIL import ImageFile
import glob
import numpy as np
from pathlib import Path
from csv import DictReader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from .utils import DataLoader

class FashionDataLoader:

    def __init__(
            self,
            base_path: str,
            target_size: Tuple[int, int] = (80,60)
    ) -> None:
        """
        Initialize the FashionDataLoader.

        This constructor sets up the base directory for the dataset and defines
        the target image size used when loading and preprocessing images. The
        base path is expected to contain the required dataset structure,
        including the image files and the corresponding metadata CSV.

        :param base_path: Base directory containing the dataset files, including
            the ``images`` folder and the ``styles.csv`` metadata file.
        :type base_path: str

        :param target_size: Desired image dimensions (width, height) to which
            all images will be resized during loading.
        :type target_size: Tuple[int, int]

        :return: None
        :rtype: None
        """

        self.base_path: Path = Path(base_path)
        self.target_size: Tuple[int, int] = target_size


    def load_and_split(
            self,
            seed: int = 42
    ) -> Tuple:
        """
        Load the dataset, preprocess images and labels, and split into subsets.

        This function reads image files and their corresponding metadata,
        filters the dataset based on predefined criteria (such as article type,
        gender, and usage), loads and preprocesses the images, and encodes the
        labels. The resulting dataset is then split into training and testing
        subsets using a fixed random seed for reproducibility.

        :param seed: Random seed used to ensure reproducible train–test splits.
        :type seed: int

        :return: A tuple containing the split datasets, typically in the form
            ``(X_train, X_test, y_train, y_test)``, where images are returned
            as NumPy arrays and labels are encoded accordingly.
        :rtype: Tuple
        """

        style_path: Path = self.base_path / "styles.csv"
        images_path_pattern: Path = (
            self.base_path / "images" / "*.jpg"
        )

        image_paths: List[Path] = [
            Path(each_path) for each_path in [*glob.glob(str(images_path_pattern))]
        ]

        # We focus only `Watches` Images for "Casual", "Smart Casual", and "Sports" usages
        article_type: str = "Watches"
        genders: Set = {"Women", "Men"}
        usages: Set = {"Casual", "Smart Casual", "Sports"}
        print(
            f"reading images and labels for {len(image_paths)} images from {self.base_path},\nfiltering only {article_type} for {genders} gender and {usages} usages"
        )

        with open(style_path, mode="r") as style_file_handle:
            dict_reader: DictReader = DictReader(style_file_handle)

            styles_ : List[Dict] =  [*dict_reader]

            style_dict: Dict = {
                each_style["id"]: each_style
                for each_style in styles_
                if (
                    each_style["articleType"] == article_type
                    and
                    each_style["usage"] in usages
                    and
                    each_style["gender"] in genders
                )
            }

        article_image_paths: List[Path] = [
            *filter(
                lambda pth: str(pth).split(os.path.sep)[-1][:-4] in style_dict.keys(),
                image_paths
            )
        ]

        images_, labels_ = self.load_images_and_labels(
            image_paths = article_image_paths,
            styles = style_dict,
            target_size = self.target_size
        )

        # Normalize the images, and multi-hot encode the labels
        print(f"normalizing {len(article_image_paths)} {article_type} images and convert labels into multi-hot encoded")
        images_ = images_.astype("float32") / 255.0
        mlb: MultiLabelBinarizer = MultiLabelBinarizer()
        labels_ = mlb.fit_transform(labels_)

        # Create train, test, and validation splits
        print(f"splitting {len(article_image_paths)} {article_type} images and labels into train, test, and validation sets")
        X_train, X_test, y_train, y_test = train_test_split(
            images_, labels_, stratify=labels_, test_size=0.2, random_state=seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, stratify=y_train, test_size=0.2, random_state=seed
        )

        return X_train, X_val, X_test, y_train, y_val, y_test, mlb.classes_


    def load_images_and_labels(
            self,
            image_paths: List[Path],
            styles: Dict,
            target_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a dataset of images and their corresponding labels.

        This function loads images from a list of file paths, converts each image
        into a NumPy array, and extracts labels based on the provided `styles`
        metadata. The output is a tuple of NumPy arrays containing the image data
        and their associated labels.

        :param image_paths: List of file paths to the image files.
        :type image_paths: List[str]

        :param styles: Dictionaries, each describing metadata such as
            gender and style for the corresponding image.
        :type styles: Dict[str, Dict]

        :param target_size: Desired image dimensions (width, height) for resizing.
        :type target_size: Tuple[int, int]

        :return: A tuple ``(images, labels)`` where:
            - ``images`` is a NumPy array of shape ``(n_samples, height, width, channels)``
              containing the image data.
            - ``labels`` is a NumPy array containing the corresponding label values.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """

        images: List[np.ndarray] = []
        labels: List[Tuple] = []

        for each_image_path in image_paths:

            img_image: ImageFile = load_img(
                path=each_image_path,
                target_size=target_size,
            )

            img_arr: np.ndarray = img_to_array(img_image)
            image_id = str(each_image_path).split(os.path.sep)[-1][:-4]

            img_style: Dict[str, Dict] = styles[image_id]

            labels_tpl: Tuple = (
                img_style["gender"],
                img_style["usage"],
            )

            images.append(img_arr)
            labels.append(labels_tpl)

        return np.array(images), np.array(labels)


def get_loader(
        base_path: str
) -> DataLoader:
    return FashionDataLoader(
        base_path = base_path,
    )