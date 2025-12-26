from typing import Dict

from pathlib import Path
import os
import sys

import pandas as pd
import yaml
import kagglehub
from kagglehub import KaggleDatasetAdapter

CUR_DIRECTORY: Path = Path(__file__).parent
ROOT_DIRECTORY: Path = CUR_DIRECTORY.parent
DATASET_DIRECTORY: Path = ROOT_DIRECTORY / "datasets"
GCP_SERVICE_ACCOUNT_PATH: Path = ROOT_DIRECTORY / "credentials" / "gcp" / "service_account.json"
KAGGLE_JSON_PATH: Path = ROOT_DIRECTORY / "kaggle" / "kaggle.json"


def authenticate(

) -> None:

    if "google.colab" in sys.modules:
        pass
    else:
        # Local PyCharm
        pass


