from typing import Optional,  Union, Any, Callable
import tempfile
import traceback

from pathlib import Path

from google.cloud.storage import Client, Bucket, Blob
from google.cloud.exceptions import NotFound, Conflict, GoogleCloudError

from ..GCP.CredentialAccessor import CredentialAccessor


class DaoCloudStorage:
    """
        Data Access Object (DAO) for interacting with Google Cloud Storage (GCS).

        This class provides a high-level, idempotent interface for common GCS
        operations, including creating a client, managing buckets, and performing
        file (blob) operations such as upload, download, and deletion.

        The class relies on a :class:`.CredentialAccessor` instance to obtain
        GCP service account credentials, the default project ID, and the target
        location for GCS operations.

        Parameters
        ----------
        credential_accessor : CredentialAccessor
            An object that provides access to the GCP service account credentials
            and project ID used for GCS interactions.

        Attributes
        ----------
        credential_accessor : CredentialAccessor
            Reference to the credential accessor provided during initialization.
        mygcs : google.cloud.storage.client.Client
            GCS client used to perform all interactions with Google Cloud Storage.

        Examples
        --------
        >>> from pathlib import Path
        >>> dao = DaoCloudStorage(credential_accessor)
        >>> local_file = Path("/tmp/data.csv")
        >>> dao.upload_blob_from_file(
        ...     bucket_name="my-data-lake-raw",
        ...     object_name="source/api/data.csv",
        ...     file_path=local_file
        ... )
        <Blob object>
        """

    def __init__(
            self,
            credential_accessor: CredentialAccessor
    ) -> None:

        self.credential_accessor: CredentialAccessor = credential_accessor
        self.mygcs: Client = self.create_client()

    def create_client(
            self
    ) -> Client:

        """
        Create a Google Cloud Storage client using the provided credentials.

        Returns
        -------
        google.cloud.storage.client.Client
            A configured GCS client instance.

        Notes
        -----
        The client is created using the service account credentials and project
        ID exposed by :attr:`credential_accessor`. This method is invoked
        internally during initialization.
        """

        return Client(
            credentials=self.credential_accessor.gcp_sa_credentials,
            project=self.credential_accessor.project_id
        )

    def is_exists(
            self,
            bucket_name: str,
            object_name: Optional[str] = None
    ) -> bool:

        """
        Check whether a GCS bucket or a specific object (blob) exists.

        This method supports checking for the existence of a bucket alone,
        or an object within a valid bucket.

        Parameters
        ----------
        bucket_name : str
            Name of the GCS bucket to check.
        object_name : str, optional
            Full path/key of the object (blob) within the bucket. If ``None``,
            only the bucket's existence is checked.

        Returns
        -------
        bool
            ``True`` if the specified bucket/object exists, ``False`` otherwise.
        """

        print(f"Checking if bucket `{bucket_name}` exists...")

        try:

            _bucket: Bucket = self.mygcs.get_bucket(bucket_name)

            # Case 1: Only check for bucket existence
            if object_name is None:
                print(f"Bucket `{bucket_name}` exists.")
                return True

            _blob: Blob = _bucket.get_blob(object_name)

            # Case 2: Checking for object existence in bucket
            if _blob:
                print(f"Object `{object_name}` exists in bucket `{bucket_name}`.")
                return True

            # If we reach here, the bucket exists but the object does not
            print(f"Object `{object_name}` does not exist in bucket `{bucket_name}`.")
            return False

        except NotFound:
            print(f"Bucket `{bucket_name}` does not exist.")
            return False

        except GoogleCloudError:
            print(f"Error occurred while checking if bucket `{bucket_name}` exists.")
            traceback.print_exc()
            return False


    def create_bucket(
            self,
            bucket_name: str,
    ) -> Optional[Bucket]:

        """
        Create a new GCS bucket.

        This method is designed to be **idempotent**: if the bucket already
        exists (a :class:`~google.cloud.exceptions.Conflict` occurs), it
        retrieves and returns the existing bucket object instead of raising an
        error.

        Parameters
        ----------
        bucket_name : str
            The unique name for the new bucket.

        Returns
        -------
        google.cloud.storage.bucket.Bucket or None
            The created or existing Bucket object upon success, or ``None``
            if a non-recoverable :class:`~google.cloud.exceptions.GoogleCloudError`
            occurs.
        """

        print(f"Attempting to create bucket `{bucket_name}`...")

        # 1. Check if the bucket_name already exists. If it does, return the bucket object.
        if self.is_exists(bucket_name=bucket_name):
            try:
                _bucket: Bucket = self.mygcs.get_bucket(bucket_or_name=bucket_name)
                print(f"Bucket `{bucket_name}` already exists. Returning bucket object.")
                return _bucket
            except NotFound:
                pass

        try:
            _bucket: Bucket = self.mygcs.create_bucket(
                bucket_or_name=bucket_name,
                location=self.credential_accessor.location
            )

            print(f"Bucket `{bucket_name}` successfully created.")
            return _bucket

        except Conflict:
            print(f"Bucket `{bucket_name}` already exists. (Conflict Detected)")
            return self.mygcs.get_bucket(bucket_or_name=bucket_name)

        except GoogleCloudError:
            print(f"Error occurred while creating bucket `{bucket_name}`.")
            traceback.print_exc()
            return None


    def upload_blob_from_file(
            self,
            bucket_name: str,
            object_name: str,
            file_path: Union[str, Path]
    ) -> Optional[Blob]:

        """
        Upload a file from the local filesystem to a GCS object (blob).

        This method will check for the bucket's existence and attempt to
        create it if it does not exist (leveraging :meth:`create_bucket`).
        If the object already exists, it will be overwritten.

        Parameters
        ----------
        bucket_name : str
            Name of the target GCS bucket.
        object_name : str
            The full key or path of the object within the bucket
            (e.g., 'data/raw/file.csv').
        file_path : str or :class:`pathlib.Path`
            The full local path to the file to be uploaded.

        Returns
        -------
        google.cloud.storage.blob.Blob or None
            The uploaded Blob object upon success, or ``None`` if the local file
            is not found or a cloud error occurs.

        Raises
        ------
        Exception
            If the necessary bucket cannot be retrieved or created, indicating
            a critical setup failure.
        """

        if isinstance(file_path, str):
            local_path = Path(file_path)
        else:
            local_path = file_path

        if not local_path.is_file():
            print(f"🚫 ERROR: Local File Not found at path `{local_path}`.")
            return None

        print(f"Attempting to upload file `{local_path}` to bucket `{bucket_name}`...")

        bucket_obj: Optional[Bucket] = None

        if not self.is_exists(bucket_name=bucket_name):
            print(f"Bucket `{bucket_name}` does not exist. Attempting to create it...")
            # create_bucket handles Conflict and returns the Bucket object if successful
            bucket_obj = self.create_bucket(bucket_name=bucket_name)
        else:
            # If it exists, retrieve the object reference
            try:
                bucket_obj = self.mygcs.get_bucket(bucket_or_name=bucket_name)
            except NotFound:
                # Should not happen after is_exists, but is defensive.
                raise Exception(f"ERROR: Could not retrieve existing bucket `{bucket_name}`.")
                return None

        # Check if we failed to get/create the bucket
        if bucket_obj is None:
            raise Exception("🛑 ERROR: ABORT: Failed to retrieve or create the necessary bucket.")
            return None

        try:

            _blob: Blob = bucket_obj.blob(blob_name=object_name)
            _blob.upload_from_filename(
                filename=str(local_path)
            )
            print(f"File `{local_path}` successfully uploaded to bucket `{bucket_name}`.")
            return _blob

        except (GoogleCloudError, Exception):
            print(f"Error occurred while uploading file `{local_path}` to bucket `{bucket_name}`.")
            traceback.print_exc()
            return None


    def download_blob_to_file(
            self,
            bucket_name: str,
            object_name: str,
            file_path: Union[str, Path]
    ) -> Optional[Path]:

        """
        Download a GCS object (blob) to a specified local file path.

        This method performs checks for both bucket and object existence
        before attempting the download. It also ensures the necessary
        local directory structure for the target file exists.

        Parameters
        ----------
        bucket_name : str
            Name of the source GCS bucket.
        object_name : str
            The full key or path of the object in GCS to be downloaded.
        file_path : str or :class:`pathlib.Path`
            The full local path where the file should be saved.

        Returns
        -------
        :class:`pathlib.Path` or None
            The path to the newly downloaded local file upon success, or
            ``None`` if the bucket/object does not exist or an error occurs.
        """

        if isinstance(file_path, str):
            local_path = Path(file_path)
        else:
            local_path = file_path

        print(f"Attempting to download file `{object_name}` from bucket `{bucket_name}` to `{local_path}`...")

        if not self.is_exists(bucket_name=bucket_name):
            print(f"🚫 ERROR: Bucket `{bucket_name}` does not exist. Cannot download file `{object_name}`.")
            return None

        elif not self.is_exists(bucket_name=bucket_name, object_name=object_name):
            print(
                f"🚫 ERROR: Object `{object_name}` does not exist in bucket `{bucket_name}`. Cannot download.")
            return None

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            print(f"🚫ERROR: Failed to create neccessary local directory for file `{local_path}`.")
            traceback.print_exc()
            return None

        try:
            _bucket: Bucket = self.mygcs.get_bucket(bucket_or_name=bucket_name)
            _blob: Blob = _bucket.blob(blob_name=object_name)
            _blob.download_to_filename(filename=str(local_path))

            print(f"File `{object_name}` successfully downloaded from bucket `{bucket_name}` to `{local_path}`.")
            return local_path

        except GoogleCloudError:
            print(f"🚫ERROR: Error occurred while downloading file `{object_name}` from bucket `{bucket_name}`.")
            traceback.print_exc()
            return None

        except Exception as exc:
            print(f"🚫ERROR: Unexpected error occurred while downloading file `{object_name}` from bucket `{bucket_name}`.")
            traceback.print_exc()
            return None



    def process_blob_with_handler(
            self,
            bucket_name: str,
            object_name: str,
            callback_func: Callable,
            *args,
            **kwargs
    ) -> Optional[Any]:
        """
        Downloads a GCS object to a temporary local file and passes the file path
        to a handler function for processing. The temporary file is guaranteed
        to be deleted immediately after the handler returns or raises an error.

        Parameters
        ----------
        bucket_name : str
            Name of the source GCS bucket.
        object_name : str
            The full key or path of the object in GCS to be downloaded.
        handler_func : callable
            A function that accepts the downloaded temporary file's
            :class:`pathlib.Path` object as its first argument, plus any
            additional positional or keyword arguments.
        *args : tuple, optional
            Positional arguments to pass to the handler_func.
        **kwargs : dict, optional
            Keyword arguments to pass to the handler_func.

        Returns
        -------
        Any or None
            The result returned by the ``handler_func`` upon success, or ``None``
            if the download itself failed.
        """

        print(f"Attempting to download file `{object_name}` from bucket `{bucket_name}`...")

        if not self.is_exists(bucket_name=bucket_name):
            print(f"🚫 ERROR: Bucket `{bucket_name}` does not exist. Cannot download file `{object_name}`.")
            return None

        elif not self.is_exists(bucket_name=bucket_name, object_name=object_name):
            print(
                f"🚫 ERROR: Object `{object_name}` does not exist in bucket `{bucket_name}`. Cannot download.")
            return None

        try:
            with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                temp_file_path: Path = Path(tmp_file.name)
                tmp_file.close() # close the FileHandler so GCS can write to it

                _bucket: Bucket = self.mygcs.get_bucket(bucket_or_name=bucket_name)
                _blob: Blob = _bucket.blob(blob_name=object_name)
                _blob.download_to_filename(filename=str(temp_file_path))

                print(f"File `{object_name}` successfully downloaded from bucket `{bucket_name}` to temporary file `{temp_file_path}`.")

                print(f"Processing blob with Callback Function `{callback_func.__name__}`...")

                result: Any = callback_func(temp_file_path, *args, **kwargs)

                print(f"Processing completed for {object_name}.")
                return result


        except NotFound:
            print(f"🚫 ERROR: Object `{object_name}` or bucket `{bucket_name} does not exists. Cannot download.")
            traceback.print_exc()
            return None

        except GoogleCloudError:
            print(f"🚫 ERROR: Failed to download file `{object_name}` from bucket `{bucket_name}`.")
            traceback.print_exc()
            return None

        except Exception as exc:
            print(f"🚫 ERROR: Unexpected error occurred while downloading file `{object_name}` from bucket `{bucket_name}`.")
            traceback.print_exc()
            return None



    def delete_blob(
            self,
            bucket_name: str,
            object_name: str
    ) -> Optional[bool]:

        """
        Delete a specific GCS object (blob).

        This method is **idempotent**: if the object or its container bucket
        does not exist, the function returns success without error, as the
        desired state (object removed) is already achieved.

        Parameters
        ----------
        bucket_name : str
            Name of the GCS bucket containing the object.
        object_name : str
            The full key or path of the object in GCS to be deleted.

        Returns
        -------
        bool or None
            ``True`` if the object was successfully deleted or was already
            absent (idempotency), or ``None`` if a cloud error prevents the
            operation.
        """

        print(f"Attempting to delete object `{object_name}` from bucket `{bucket_name}`...")

        if not self.is_exists(bucket_name=bucket_name):
            print(f"🚫 ERROR: Bucket `{bucket_name}` does not exist. Cannot delete object `{object_name}`.")
            return True # Idempotency: Object does not exists, since the bucket does not exist.

        elif not self.is_exists(bucket_name=bucket_name, object_name=object_name):
            print(f"🚫 ERROR: Object `{object_name}` does not exist in bucket `{bucket_name}`. Cannot delete.")
            return True # Idempotency: Object does not exists, since the object itself does not exist.

        try:
            _bucket: Bucket = self.mygcs.get_bucket(bucket_or_name=bucket_name)
            _blob: Blob = _bucket.blob(blob_name=object_name)
            _blob.delete()

            print(f"Object `{object_name}` successfully deleted from bucket `{bucket_name}`.")

            return True

        except GoogleCloudError:
            print(f"🚫 ERROR: Error occurred while deleting object `{object_name}` from bucket `{bucket_name}`.")
            traceback.print_exc()
            return None

        except Exception as exc:
            print(f"🚫ERROR: Unexpected error occurred while deleting object `{object_name}` from bucket `{bucket_name}`.")
            traceback.print_exc()
            return None