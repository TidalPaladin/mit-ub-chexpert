from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from torch.utils.data import DataLoader


try:
    import pytorch_lightning  # noqa: F401
except ImportError:
    raise ImportError(
        "The 'train' extra must be installed to use this module. " "Install with 'pip install mit-ub-chexpert[train]'"
    )


import torchvision.transforms.v2 as tv2
from torch_dicom.datasets import collate_fn
from torch_dicom.datasets.helpers import Transform
from torch_dicom.datasets.image import ImagePathDataset, save_image
from torch_dicom.preprocessing import MinMaxCrop, Resize
from tqdm_multiprocessing import ConcurrentMapper


def _preprocess_image(example: Dict[str, Any], root: Path, dest: Path, compression: str | None = None):
    image = example["img"].squeeze()
    source = example["path"][0]
    dest_path = dest / source.relative_to(root)
    dest_path = dest_path.with_suffix(".tiff")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # Source images are 8 bit grayscale JPEG so we save in 8 bits
    save_image(image, dest_path, dtype=cast(Any, np.uint8), compression=compression)


def preprocess_chexpert(
    source: Path,
    dest: Path,
    size: Tuple[int, int] | None = None,
    num_workers: int = 8,
):
    if not source.is_dir():
        raise NotADirectoryError(source)
    if not dest.is_dir():
        raise NotADirectoryError(dest)

    # Prepare crop and resize
    transforms: List[Transform] = [
        MinMaxCrop(),
    ]
    if size is not None:
        transforms.append(Resize(size, mode="max"))
    transform = tv2.Compose(transforms)

    # Prepare dataset and dataloader
    sources = source.rglob("*.jpg")
    ds = ImagePathDataset(sources, transform=transform)
    dl = DataLoader(ds, num_workers=num_workers, batch_size=1, collate_fn=collate_fn)

    with ConcurrentMapper(jobs=num_workers) as mapper:
        mapper.create_bar(total=len(ds), desc="Preprocessing images")
        func = partial(_preprocess_image, root=source, dest=dest, compression="packbits")
        mapper(func, dl)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Preprocess CheXpert dataset")
    parser.add_argument("source", type=Path, help="Source directory")
    parser.add_argument("dest", type=Path, help="Destination directory")
    parser.add_argument("--size", type=int, nargs=2, default=None, help="Size of the images")
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def main(args: Namespace):
    preprocess_chexpert(args.source, args.dest, args.size, args.num_workers)


def entrypoint():
    main(parse_args())


if __name__ == "__main__":
    entrypoint()
