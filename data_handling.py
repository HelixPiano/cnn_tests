import lightning as L
from torch.utils.data import DataLoader
import xarray as xr
import torch
import numpy as np


class DataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.data_train = None
        self.data_test = None
        self.data_val = None

    def setup(self, stage: str):
        gwl_train, gwl_test, indices_test, indices_training = load_gwl()
        geo_train, geo_test = load_data("C:/Users/Philipp/PycharmProjects/cnn/ERA20/daily/129_r/129.nc", 129,
                                        indices_test, indices_training)
        mslp_train, mslp_test = load_data("C:/Users/Philipp/PycharmProjects/cnn/ERA20/daily/151_r/151.nc", 151,
                                          indices_test, indices_training)

        train_data = torch.stack([geo_train, mslp_train], dim=1)
        test_data = torch.stack([geo_test, mslp_test], dim=1)

        # Assign train/val datasets for use in dataloaders
        self.data_train = torch.utils.data.TensorDataset(train_data, gwl_train)

        self.data_val = torch.utils.data.TensorDataset(test_data, gwl_test)

        self.data_test = torch.utils.data.TensorDataset(test_data, gwl_test)

    def train_dataloader(self):
        print(self.data_train)
        return DataLoader(self.data_train, batch_size=32, num_workers=1, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=32, num_workers=1, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=32)


def load_gwl():
    ds = xr.load_dataset("C:/Users/Philipp/PycharmProjects/cnn/Wetterlagen/GWL_Hess_Brezowsky_1881-2022.nc")
    gwl_train = ds.sel(time=slice('1900-01-01', '1969-12-31')).GWL.to_numpy()
    gwl_train -= 1
    gwl_test = ds.sel(time=slice('1970-01-01', '1979-12-31')).GWL.to_numpy()
    gwl_test -= 1
    indices_test = np.nonzero(gwl_test != 29)
    indices_training = np.nonzero(gwl_train != 29)
    return torch.LongTensor(gwl_train[indices_training]), torch.LongTensor(
        gwl_test[indices_test]), indices_test, indices_training


def load_data(input_path, var, indices_test, indices_training):
    data = xr.load_dataset(input_path)[f"var{var}"]
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    data_train = data.sel(time=slice('1900-01-01', '1969-12-31')).to_numpy()
    data_test = data.sel(time=slice('1970-01-01', '1979-12-31')).to_numpy()
    return torch.from_numpy(data_train[indices_training].astype(np.float32)), torch.from_numpy(
        data_test[indices_test].astype(np.float32))
