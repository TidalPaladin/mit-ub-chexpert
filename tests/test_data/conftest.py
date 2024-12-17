from pathlib import Path
from typing import List

import pandas as pd
import pytest
from PIL import Image


@pytest.fixture(scope="module")
def image_factory():
    def func(root: Path, train: bool, patient: str, study: str, filename: str, as_tiff: bool = False) -> Path:
        path = (
            root
            / ("train" if train else "valid")
            / f"patient{patient}"
            / f"study{study}"
            / f"{filename}.{'tiff' if as_tiff else 'jpg'}"
        )
        H, W = 128, 128
        img = Image.new("L", (W, H), (255,))
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return path

    return func


@pytest.fixture(scope="module")
def csv_factory():
    def func(root: Path, train: bool, patients: List[str], studies: List[str], filenames: List[str]) -> Path:
        header = [
            "Path",
            "Sex",
            "Age",
            "Frontal/Lateral",
            "AP/PA",
            "No Finding",
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
        ]

        # Values can be 1.0, 0.0, -1.0 or missing. -1.0 indicates unknown. Authors treat missing as 0.0
        data = []
        for patient, study, filename in zip(patients, studies, filenames):
            # Randomly choose values for this row
            row = [
                f"CheXpert-v1.0/{'train' if train else 'valid'}/patient{patient}/study{study}/{filename}.jpg",
                "Female" if int(patient) % 2 == 0 else "Male",
                str(20 + int(patient) % 60),
                "Frontal" if int(study) % 2 == 0 else "Lateral",
                "AP" if int(study) % 2 == 0 else "PA",
                *[str([-1.0, 0.0, 1.0, ""][int(patient) % 4]) for i in range(len(header) - 5)],
            ]
            data.append(row)
        df = pd.DataFrame(data, columns=header)
        csv_path = root / ("train.csv" if train else "valid.csv")
        df.to_csv(csv_path, index=False)
        return csv_path

    return func


@pytest.fixture
def data_factory(tmp_path, csv_factory, image_factory):
    def func(train: bool, as_tiff: bool = False):
        patients = [0, 1, 2]
        studies = [1, 2, 1]
        filenames = ["view1_frontal", "view1_frontal", "view1_frontal"]

        csv_path = csv_factory(tmp_path, train=train, patients=patients, studies=studies, filenames=filenames)
        for patient, study, filename in zip(patients, studies, filenames):
            image_factory(tmp_path, train=train, patient=patient, study=study, filename=filename, as_tiff=as_tiff)
        return csv_path

    return func
