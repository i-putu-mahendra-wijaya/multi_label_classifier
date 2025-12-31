from pathlib import Path
import os
import shutil

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