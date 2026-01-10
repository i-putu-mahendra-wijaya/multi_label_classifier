from typing import Protocol, Any, Dict, Set, List, Tuple, Any, runtime_checkable

import os
import argparse
from argparse import Namespace
from pathlib import Path
import glob

from PIL import ImageFile
import numpy as np
from csv import DictReader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


############################################################################
#
# utils
# Defining Protocol (Interface)
#
############################################################################

@runtime_checkable
class DeepLearningModel(Protocol):

    def fit(
            self,
            x: Any,
            y: Any,
            validation_data: Tuple,
            epochs: int,
            batch_size: int,
            **kwargs,
    ) -> Any:
        ...

    def save(
            self,
            filepath: str
    ) -> None:
        ...

    def evaluate(
            self,
            x: Any,
            y: Any,
            batch_size: int,
    ) -> Tuple[float, float]:
        ...

    def summary(
            self
    ) -> None:
        ...

    def predict(
            self,
            x: Any
    ) -> np.ndarray:
        ...


@runtime_checkable
class DataLoader(Protocol):

    def load_and_split(
            self,
            seed: int,
    ) -> Tuple: # Returns X_train, X_val, X_test, y_train, y_val, y_test, mlb_classes
        ...


############################################################################
#
# data_loader
# Defining Data Loader helper class
#
############################################################################

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


############################################################################
#
# model
# Defining CNN Model
#
############################################################################

class MultiLabelCNN:

    def __init__(
            self,
            learning_rate: float,
            width: int,
            height: int,
            depth: int,
            classes: int
    ) -> None:
        """
        Build a Convolutional Neural Network (CNN) model for multi-label classification.

        This function constructs a CNN architecture tailored for multi-label classification tasks.
        The model expects input images with specified width, height, and depth (number of channels),
        and produces output predictions across a defined number of label classes.

        :param learning_rate: Learning rate for the model.
        :type learning_rate: float

        :param width: Width of the input image in pixels.
        :type width: int

        :param height: Height of the input image in pixels.
        :type height: int

        :param depth: Number of color channels in the input image (e.g., 1 for grayscale, 3 for RGB).
        :type depth: int

        :param classes: Number of label classes to predict.
        :type classes: int
        """

        self.learning_rate: float = learning_rate
        self.width: int = width
        self.height: int = height
        self.depth: int = depth
        self.classes: int = classes

        self.model: Model = self.build_network(
            width=self.width,
            height=self.height,
            depth=self.depth,
            classes=self.classes
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    @staticmethod
    def build_network(
            width: int,
            height: int,
            depth: int,
            classes: int
    ) -> Model:

        input_layer: Any = Input(shape=(height, width, depth))
        conv_1: Conv2D = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
        )(input_layer)
        activation_1: ReLU = ReLU()(conv_1)
        batch_norm_1: BatchNormalization = BatchNormalization(axis=-1)(activation_1)

        conv_2: Conv2D = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
        )(batch_norm_1)
        activation_2: ReLU = ReLU()(conv_2)
        batch_norm_2: BatchNormalization = BatchNormalization(axis=-1)(activation_2)

        max_pool_1: MaxPooling2D = MaxPooling2D(
            pool_size=(2, 2)
        )(batch_norm_2)
        dropout_1: Dropout = Dropout(rate=0.25)(max_pool_1)

        conv_3: Conv2D = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
        )(dropout_1)
        activation_3: ReLU = ReLU()(conv_3)
        batch_norm_3: BatchNormalization = BatchNormalization(axis=-1)(activation_3)

        conv_4: Conv2D = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
        )(batch_norm_3)
        activation_4: ReLU = ReLU()(conv_4)
        batch_norm_4: BatchNormalization = BatchNormalization(axis=-1)(activation_4)

        max_pool_2: MaxPooling2D = MaxPooling2D(
            pool_size=(2, 2)
        )(batch_norm_4)
        dropout_2: Dropout = Dropout(rate=0.25)(max_pool_2)

        flatten_1: Flatten = Flatten()(dropout_2)
        dense_1: Dense = Dense(
            units=512
        )(flatten_1)
        activation_5: ReLU = ReLU()(dense_1)
        batch_norm_5: BatchNormalization = BatchNormalization(axis=-1)(activation_5)
        dropout_3: Dropout = Dropout(rate=0.5)(batch_norm_5)

        dense_4: Dense = Dense(
            units=classes
        )(dropout_3)

        output: Activation = Activation("sigmoid")(dense_4)

        return Model(
            inputs=input_layer
            , outputs=output
        )

    def fit(
            self,
            X_train: Any,
            y_train: Any,
            validation_data: Tuple,
            epochs: int,
            batch_size: int,
            *args,
            **kwargs,
    )-> Any:
        return self.model.fit(
            X_train,
            y_train,
            validation_data = validation_data,
            epochs = epochs,
            batch_size = batch_size,
            verbose = 2,
            *args,
            **kwargs
        )

    def save(
            self,
            filepath: str,
    ) -> None:
        self.model.save(
            filepath = filepath
        )

    def evaluate(
            self,
            X_test: Any,
            y_test: Any,
            batch_size: int,
    ) -> Tuple[float, float]:
        return self.model.evaluate(X_test, y_test, batch_size=batch_size)


    def predict(
            self,
            x: Any
    ) -> np.ndarray:
        return self.model.predict(x)

    def summary(
            self,
    ) -> None:
        self.model.summary()


def create_model(
        learning_rate: float,
        width: int,
        height: int,
        depth: int,
        classes: int,
) -> DeepLearningModel:
    return MultiLabelCNN(
        learning_rate=learning_rate,
        width=width,
        height=height,
        depth=depth,
        classes=classes,
    )


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(Path(__file__).parent / "datasets" / "fashion-product-images-small")
    )
    args: Namespace = parser.parse_args()

    # 1. Define the Model Directory
    model_dir: str = os.getenv("AIP_MODEL_DIR", "../deploy/saved_models")

    if not model_dir.startswith("gs://"):
        os.makedirs(model_dir, exist_ok=True)

    # 2. Load Data via Protocol
    loader: DataLoader = get_loader(
        base_path=args.data_path,
    )
    X_train, X_val, X_test, y_train, y_val, y_test, classes = loader.load_and_split(seed=42)

    # 3. Build Model via Protocol
    cnn_model: DeepLearningModel = create_model(
        learning_rate=0.001,
        width=60,
        height=80,
        depth=3,
        classes=len(classes),
    )

    cnn_model.summary()

    # 4. Train
    cnn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # 5. Evaluate and Save
    loss, acc = cnn_model.evaluate(X_test, y_test, batch_size=args.batch_size)
    print(f"Test Accuracy: {acc:.2f}")
    cnn_model.save(f"{model_dir}/multi_label_classifier.keras")

if __name__ == "__main__":
    main()