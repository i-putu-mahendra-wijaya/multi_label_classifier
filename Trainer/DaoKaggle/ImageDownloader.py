from typing import Union

from pathlib import Path
import os

from .CredentialAccessor import CredentialAccessor


class ImageDownloader:

    def __init__(
            self,
            credential_accessor: CredentialAccessor,
    ) -> None:

        self.credential_accessor: CredentialAccessor = credential_accessor


    def download_image(
            self,
            image_dataset_url: str,
            output_dir: Union[str, Path]
    ) -> None:

        # check if output directory exists,if not, then create one
        output_dir.mkdir(parents=True, exist_ok=True)

        os.environ["KAGGLE_CONFIG_DIR"] = str(self.credential_accessor.kaggle_json_path)
        os.environ["KAGGLEHUB_CACHE_DIR"] = str(output_dir)

        import kagglehub

        _downloaded_dataset_path: str = kagglehub.dataset_download(
            handle = image_dataset_url,
        )

