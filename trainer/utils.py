from typing import Protocol, Any, Tuple, Any, runtime_checkable
from pathlib import Path
import os
import shutil

import numpy as np
import tensorflow as tf

def create_symlink(
        source: Path,
        link_path: Path,
        overwrite: bool = True,
) -> Path:

    source: Path = source.expanduser().absolute()
    link_path: Path = link_path.expanduser().absolute()

    if source == link_path:
        raise ValueError(f"Refusing to create a symlink loop: source == link_path {source}")

    # If current link_path already points to a source, then just return the existing link_path
    if link_path.is_symlink():
        try:
            current_target: Path = (link_path.parent / link_path.readlink()).absolute()
        except OSError:
            current_target = None

        if current_target is not None and current_target == source:
            return link_path

    if os.path.lexists(link_path):
        if not overwrite:
            raise FileExistsError(f"{link_path} already exists")

        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        else:
            shutil.rmtree(link_path)

    link_path.symlink_to(
        target = source,
        target_is_directory = source.is_dir(),
    )

    return link_path


"""
Herein below are definitions of the Protocols used in this model. 
We are going to use `interface-driven design` as far as possible in this project
"""

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