from typing import Dict, Optional
from pathlib import Path

import json

class CredentialAccessor:
    def __init__(
            self,
            kaggle_json_path: Path,
            kaggle_api_token_path: Optional[Path] = None
    ) -> None:

        self.kaggle_json_path: Path = kaggle_json_path
        self.kaggle_api_token_path: Optional[Path] = kaggle_api_token_path

        self.kaggle_json: Dict = {}
        self.kaggle_api_token: Dict = {}

        with open(self.kaggle_json_path, "r") as kjf:
            _json_string: str = kjf.read()
            self.kaggle_json: Dict = json.loads(_json_string)

        if kaggle_api_token_path is not None:
            with open(self.kaggle_api_token_path, "r") as katf:
                _json_string: str = katf.read()
                self.kaggle_api_token: Dict = json.loads(_json_string)