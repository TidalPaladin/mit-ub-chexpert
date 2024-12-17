from abc import ABC
from copy import copy
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

from lightning_fabric.utilities.rank_zero import rank_zero_info
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch_dicom.datasets import collate_fn
from torch_dicom.datasets.helpers import Transform

from .dataset import CheXpert


class ChexpertDataModule(LightningDataModule, ABC):
    r"""Chexpert data module.

    .. note::
        It is recommended to use `torchvision.transforms.v2` for transforms.

    Args:
        batch_size: Size of the batches.
        seed: Seed for random number generation.
        train_transforms: Transformations to apply to the training images.
        train_gpu_transforms: GPU transformations to apply to the training images.
        val_transforms: Transformations to apply to the validation images.
        test_transforms: Transformations to apply to the test images.
        train_dataset_kwargs: Additional keyword arguments for the training dataset.
        dataset_kwargs: Additional keyword arguments for inference datasets.
        num_workers: Number of workers for data loading.
        pin_memory: Whether to pin memory.
        prefetch_factor: Prefetch factor for data loading.

    Keyword Args:
        Forwarded to :class:`torch.utils.data.DataLoader`.
    """

    dataset_train: Dataset
    dataset_val: Dataset
    dataset_test: Dataset

    def __init__(
        self,
        root: str | Path,
        batch_size: int = 4,
        seed: int = 42,
        train_transforms: Optional[Callable] = None,
        train_gpu_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        train_dataset_kwargs: Dict[str, Any] = {},
        dataset_kwargs: Dict[str, Any] = {},
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: Optional[int] = None,
        sequential: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.seed = seed
        self.train_transforms = cast(Transform, train_transforms)
        self.train_gpu_transforms = cast(Transform, train_gpu_transforms)
        self.val_transforms = cast(Transform, val_transforms)
        self.test_transforms = cast(Transform, test_transforms)
        self.train_dataset_kwargs = train_dataset_kwargs
        self.dataset_kwargs = dataset_kwargs
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.dataloader_config = kwargs
        self.sequential = sequential

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        match stage:
            case "fit":
                # prepare training dataset
                rank_zero_info("Preparing training datasets")
                self.dataset_train = CheXpert(
                    self.root, train=True, transform=self.train_transforms, sort_by_inodes=self.sequential
                )
                self.dataset_val = CheXpert(self.root, train=False, transform=self.val_transforms, sort_by_inodes=True)

            case None:
                pass  # pragma: no cover

            case _:
                raise ValueError(f"Unknown stage: {stage}")  # pragma: no cover

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        if not hasattr(self, "dataset_train"):
            raise RuntimeError("setup() must be called before train_dataloader()")  # pragma: no cover
        return self._data_loader(self.dataset_train, shuffle=not self.sequential)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader."""
        if not hasattr(self, "dataset_val"):
            raise RuntimeError("setup() must be called before val_dataloader()")  # pragma: no cover
        return self._data_loader(self.dataset_val, shuffle=False)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader."""
        if hasattr(self, "dataset_test"):
            return self._data_loader(self.dataset_test, shuffle=False)
        elif hasattr(self, "dataset_val"):
            return self._data_loader(self.dataset_val, shuffle=False)
        else:
            raise RuntimeError("setup() must be called before test_dataloader()")  # pragma: no cover

    def _data_loader(self, dataset: Dataset, **kwargs) -> DataLoader:
        config = copy(self.dataloader_config)
        config.update(kwargs)
        config["batch_size"] = self.batch_size

        # Torch forces us to pop these arguments when using a batch_sampler
        if config.get("batch_sampler", None) is not None:
            config.pop("batch_size", None)
            config.pop("shuffle", None)
            config.pop("sampler", None)

        return DataLoader(
            dataset,
            collate_fn=partial(collate_fn, default_fallback=False),
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            **config,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        assert self.trainer is not None
        # TODO: Should we consider allowing GPU transforms for val/test?
        # This was originally added to speed up training which is more augmentation intensive
        if self.trainer.training and self.train_gpu_transforms is not None:
            batch = self.train_gpu_transforms(batch)
        return batch
