import sys

import pytest
from torch.testing import assert_close

from mit_ub_chexpert.data.dataset import load_image
from mit_ub_chexpert.data.preprocess import entrypoint


@pytest.mark.parametrize("size", [None, (32, 32)])
def test_preprocess_chexpert(tmp_path, data_factory, size):
    output = tmp_path / "output"
    output.mkdir()
    data_factory(train=True)
    data_factory(train=False)
    sys.argv = ["mit_ub.data.chexpert", str(tmp_path), str(output), "--num_workers", "0"]
    if size is not None:
        sys.argv.extend(["--size", str(size[0]), str(size[1])])

    entrypoint()

    for source_path in tmp_path.rglob("*.jpg"):
        relpath = source_path.relative_to(tmp_path)
        dest_path = (output / relpath).with_suffix(".tiff")
        assert dest_path.is_file()
        if size is None:
            source = load_image(source_path)
            dest = load_image(dest_path)
            assert_close(source, dest)
