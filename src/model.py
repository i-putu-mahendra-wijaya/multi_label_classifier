from typing import Any, Tuple

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from .utils import DeepLearningModel

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

        input_layer: Input = Input(shape=(height, width, depth))
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