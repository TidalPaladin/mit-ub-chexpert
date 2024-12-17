import pytest

from mit_ub_chexpert.data.dataset import CheXpert


class TestCheXpertDataset:

    @pytest.mark.parametrize("as_tiff", [False, True])
    def test_train(self, tmp_path, data_factory, as_tiff):
        data_factory(train=True, as_tiff=as_tiff)
        dataset = CheXpert(tmp_path, train=True)
        assert len(dataset) == 3
        e1 = dataset[0]
        assert e1["img"].shape == (1, 128, 128)
        assert e1["finding"] in [-1.0, 0.0, 1.0]
        assert e1["dense_label"].shape == (13,)

    @pytest.mark.parametrize("as_tiff", [False, True])
    def test_valid(self, tmp_path, data_factory, as_tiff):
        data_factory(train=False, as_tiff=as_tiff)
        dataset = CheXpert(tmp_path, train=False)
        assert len(dataset) == 3
        e1 = dataset[0]
        assert e1["img"].shape == (1, 128, 128)
        assert e1["finding"] in [-1.0, 0.0, 1.0]
        assert e1["dense_label"].shape == (13,)
