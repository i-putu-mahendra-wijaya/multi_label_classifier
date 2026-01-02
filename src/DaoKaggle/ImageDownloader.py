from typing import Union, Dict, Optional

from pathlib import Path
import os

from .CredentialAccessor import CredentialAccessor
from ..utils import create_symlink


class ImageDownloader:

    def __init__(
            self,
            kaggle_credential_accessor: Optional[CredentialAccessor] = None,
    ) -> None:

        self.credential_accessor: Optional[CredentialAccessor] = kaggle_credential_accessor


    def download_image(
            self,
            image_dataset_url: str,
            output_dir: Union[str, Path]
    ) -> Path:

        # check if output directory exists,if not, then create one
        output_dir_path: Path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        if (
                self.credential_accessor is not None
                and
                self.credential_accessor.kaggle_json_path.exists()
        ):
            # Check if environment already have kaggle.json.swp
            os.environ["KAGGLE_CONFIG_DIR"] = str(self.credential_accessor.kaggle_json_path.parent)

            kaggle_json_dict: Dict = self.credential_accessor.kaggle_json

            os.environ["KAGGLE_USERNAME"] = kaggle_json_dict["username"]
            os.environ["KAGGLE_KEY"] = kaggle_json_dict["key"]

            import kagglehub

        else:
            # if no kaggle.json.swp is found, then ask user to login to their kaggle account
            import kagglehub

            kagglehub.auth.login()

        _downloaded_dataset_path: str = kagglehub.dataset_download(
            handle=image_dataset_url,
            force_download=True
        )

        dataset_name: str = image_dataset_url.split("/")[-1]

        symlink_path: Path = output_dir_path / dataset_name

        create_symlink(
            source=Path(_downloaded_dataset_path),
            link_path=symlink_path,
        )

        return symlink_path

