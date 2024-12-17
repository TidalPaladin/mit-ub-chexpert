from pathlib import Path
from typing import Any, Callable, Dict, Final, Tuple

import pandas as pd
import torch
from dicom_preprocessing import inode_sort, load_tiff_f32
from torch import Tensor
from torch.utils.data import Dataset
from torch_dicom.datasets.image import load_image as td_load_image


LABELS: Final = (
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)
POSITIVE_FINDING: Final = 1.0
NEGATIVE_FINDING: Final = 0.0
UNKNOWN_FINDING: Final = -1.0


def load_image(path: Path) -> Tensor:
    if path.suffix == ".tiff":
        x = load_tiff_f32(path)  # type: ignore
        x = torch.from_numpy(x).squeeze_(-1)
    else:
        x = td_load_image(path)
    return x


def label_to_tensor(label: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
    r"""Convert a dictionary of labels to a tensor of dense labels and an 'any-finding' label."""
    dense_label = torch.stack([torch.tensor(label[label_name], dtype=torch.float32) for label_name in LABELS])
    min_label, max_label = dense_label.aminmax()
    if max_label == POSITIVE_FINDING:
        finding = dense_label.new_tensor(POSITIVE_FINDING)
    elif min_label == NEGATIVE_FINDING:
        finding = dense_label.new_tensor(NEGATIVE_FINDING)
    else:
        finding = dense_label.new_tensor(UNKNOWN_FINDING)
    return finding, dense_label


class CheXpert(Dataset):
    r"""CheXpert dataset implementation.

    The root directory should contain the `train.csv` and `valid.csv` files, along with the `train` and `valid` subdirectories.
    Images may be either original CheXpert JPEGs or preprocessed TIFFs.

    Args:
        root: Root directory of the dataset.
        train: Whether to load the training or validation subset.
        transform: Transform to apply to the images.
        sort_by_inodes: Whether to sort the dataset by inode to optimize sequential read.
    """

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        sort_by_inodes: bool = True,
    ):
        super().__init__()
        self.transform = transform
        root = Path(root)
        if not root.is_dir():
            raise NotADirectoryError(root)

        # Select image subfolder
        self.root = root / ("train" if train else "valid")

        # Load metadata CSV
        csv_path = root / ("train.csv" if train else "valid.csv")
        self.df = pd.read_csv(csv_path)
        self.df.fillna(0.0, inplace=True)

        # Fix paths in CSV to match how we expect them
        self.df["Path"] = self.df["Path"].apply(lambda x: root / x.replace("CheXpert-v1.0/", ""))

        # Sort by inode if requested to optimize sequential read
        if sort_by_inodes:
            paths = [Path(p) for p in self.df["Path"]]
            sorted_paths = inode_sort(paths, bar=True)
            path_to_idx = {str(p): i for i, p in enumerate(sorted_paths)}
            self.df = self.df.iloc[[path_to_idx[str(p)] for p in self.df["Path"]]]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]

        # Select the path to load. This may be a .jpg file (original CheXpert) or a .tiff file (preprocessed)
        path = Path(row["Path"])
        if path.is_file():
            path = path
        elif path.with_suffix(".tiff").is_file():
            path = path.with_suffix(".tiff")
        else:
            raise FileNotFoundError(path)

        x = load_image(path)

        finding, dense_label = label_to_tensor(row[self.df.columns[1:]].to_dict())
        if self.transform is not None:
            x = self.transform(x)
        return {"img": x, "finding": finding, "dense_label": dense_label}
