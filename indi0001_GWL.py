"""
2024-08-09  14:56:39
"""
from datetime import datetime
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
import lightning as L
import logging
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torchmetrics as tm
import warnings
import xarray as xr


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out


class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()

        # conv unit
        self.conv_3_128 = BasicBlock(in_planes=2, planes=128)
        self.conv_128_64 = BasicBlock(in_planes=128, planes=64)
        self.conv_64_128 = BasicBlock(in_planes=64, planes=128)
        self.conv_128_128 = BasicBlock(in_planes=128, planes=128)
        self.conv_128_256 = BasicBlock(in_planes=128, planes=256)
        self.conv_256_256 = BasicBlock(in_planes=256, planes=256)
        self.conv_256_128 = BasicBlock(in_planes=256, planes=128)
        self.conv_256_64 = BasicBlock(in_planes=256, planes=64)

        # linear unit
        self.linear = nn.LazyLinear(29)

    def forward(self, x):
        out_0 = self.conv_3_128(x)
        out_1 = self.conv_128_64(out_0)
        out_2 = self.conv_64_128(out_1)
        out_3 = self.conv_128_128(out_2)
        out_4 = self.conv_128_256(out_3)
        out_5 = nn.functional.avg_pool2d(out_4, 2)
        out_6 = self.conv_256_256(out_5)
        out_7 = self.conv_256_128(out_6)
        out_8 = self.conv_128_128(out_7)
        out_9 = self.conv_128_256(out_8)
        out_10 = nn.functional.max_pool2d(out_9, 2)
        out_11 = self.conv_256_64(out_10)
        out_12 = self.conv_64_128(out_11)
        out_13 = self.conv_128_256(out_12)
        out_14 = self.conv_256_64(out_13)
        out_15 = nn.functional.max_pool2d(out_14, 2)
        out = out_15

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar


class GWLDataModule(L.LightningDataModule):
    def __init__(self, data_dir='dataset', batch_size=128, num_workers=1):
        super().__init__()
        self.data_train = None
        self.data_test = None
        self.data_val = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
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

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data_train, batch_size=self.batch_size,
                                           num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data_val, batch_size=self.batch_size,
                                           num_workers=self.num_workers, pin_memory=True, persistent_workers=True)


def training_loop() -> None:
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*Lazy modules are a new feature under heavy development*")
    warnings.filterwarnings("ignore", ".*The total number of parameters detected may*")
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    model = LightningModule()

    data_module = GWLDataModule()

    trainer = L.Trainer(max_epochs=1, fast_dev_run=False, accelerator="gpu", logger=False, precision="bf16-mixed",
                        enable_checkpointing=False,
                        callbacks=[MyProgressBar(), EarlyStopping(monitor="valid_acc", mode="max", patience=10)])

    trainer.fit(model, data_module)


class LightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = EvoCNNModel()
        self.train_acc = tm.Accuracy(task="multiclass", num_classes=29)
        self.valid_acc = tm.Accuracy(task="multiclass", num_classes=29)
        self.file_id = os.path.basename(__file__).split('.')[0]
        self.best_acc = 0

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        y_hat = self.model(inputs)
        self.train_acc(y_hat, target)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)

        loss = torch.nn.functional.cross_entropy(y_hat, target)
        self.log("my_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        y_hat = self.model(inputs)
        self.valid_acc(y_hat, target)
        self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def on_train_epoch_end(self) -> None:
        self.log_record('Train-Epoch:%3d,  Loss: %.3f, Acc:%.3f' % (
            self.current_epoch, self.trainer.logged_metrics.get("my_loss").detach(),
            self.trainer.logged_metrics.get("valid_acc").detach()))
        if self.trainer.logged_metrics.get("valid_acc").detach() > self.best_acc:
            self.best_acc = self.trainer.logged_metrics.get("valid_acc").detach()

    def on_train_end(self) -> None:
        self.log_record('Finished-Acc:%.3f' % self.best_acc)
        with open('populations/after_%s.txt' % (self.file_id[4:6]), 'a+') as f:
            f.write('%s=%.5f\n' % (self.file_id, self.best_acc))

    def log_record(self, _str):
        dt = datetime.now()
        dt.strftime('%Y-%m-%d %H:%M:%S')
        with open(f"log/{self.file_id}.txt", 'a+') as f:
            f.write('[%s]-%s\n' % (dt, _str))


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


if __name__ == '__main__':
    training_loop()
