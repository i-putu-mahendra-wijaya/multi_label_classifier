from typing import Dict

from pathlib import Path
import os

import pandas as pd
import yaml
import kagglehub
from kagglehub import KaggleDatasetAdapter

CUR_DIRECTORY: Path = Path(__file__).parent
ROOT_DIRECTORY: Path = CUR_DIRECTORY.parent
DATASET_DIRECTORY: Path = ROOT_DIRECTORY / "datasets"



