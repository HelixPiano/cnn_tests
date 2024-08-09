"""
2024-08-06  16:04:04
"""
import torch.nn as nn
import torch
import os
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import sys
from lightning.pytorch.callbacks import TQDMProgressBar
from torchvision import datasets, transforms
import torch.utils.data as tdata
import torchmetrics as tm
from datetime import datetime


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
        self.conv_3_128 = BasicBlock(in_planes=3, planes=128)
        self.conv_128_128 = BasicBlock(in_planes=128, planes=128)
        self.conv_128_64 = BasicBlock(in_planes=128, planes=64)
        self.conv_64_64 = BasicBlock(in_planes=64, planes=64)
        self.conv_64_128 = BasicBlock(in_planes=64, planes=128)
        self.conv_128_256 = BasicBlock(in_planes=128, planes=256)
        self.conv_256_256 = BasicBlock(in_planes=256, planes=256)
        self.conv_256_64 = BasicBlock(in_planes=256, planes=64)
        self.conv_64_256 = BasicBlock(in_planes=64, planes=256)

        # linear unit
        self.linear = nn.Linear(4096, 10)

    def forward(self, x):
        out_0 = self.conv_3_128(x)
        out_1 = self.conv_128_128(out_0)
        out_2 = self.conv_128_128(out_1)
        out_3 = self.conv_128_128(out_2)
        out_4 = self.conv_128_128(out_3)
        out_5 = self.conv_128_64(out_4)
        out_6 = self.conv_64_64(out_5)
        out_7 = self.conv_64_128(out_6)
        out_8 = nn.functional.avg_pool2d(out_7, 2)
        out_9 = self.conv_128_128(out_8)
        out_10 = self.conv_128_256(out_9)
        out_11 = self.conv_256_256(out_10)
        out_12 = self.conv_256_256(out_11)
        out_13 = self.conv_256_256(out_12)
        out_14 = nn.functional.avg_pool2d(out_13, 2)
        out_15 = self.conv_256_64(out_14)
        out_16 = nn.functional.avg_pool2d(out_15, 2)
        out_17 = self.conv_64_256(out_16)
        out = out_17

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


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir='dataset', batch_size=128, num_workers=1):
        super().__init__()
        self.data_train = None
        self.data_test = None
        self.data_val = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_dir, train=True, download=False)
        datasets.CIFAR10(root=self.data_dir, train=False, download=False)

    def setup(self, stage=None):
        self.data_train = datasets.CIFAR10(root=self.data_dir, train=True, transform=self.transform)
        self.data_val = datasets.CIFAR10(root=self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data_train, batch_size=self.batch_size,
                                           num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data_val, batch_size=self.batch_size,
                                           num_workers=self.num_workers, persistent_workers=True)


def training_loop() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    model = LightningModule()

    data_module = CIFAR10DataModule()

    trainer = L.Trainer(max_epochs=1, fast_dev_run=False, accelerator="gpu", logger=False, enable_checkpointing=False,
                        callbacks=[MyProgressBar(), EarlyStopping(monitor="valid_acc", mode="max", patience=10)])

    trainer.fit(model, data_module)


class LightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = EvoCNNModel()
        self.train_acc = tm.Accuracy(task="multiclass", num_classes=10)
        self.valid_acc = tm.Accuracy(task="multiclass", num_classes=10)
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
            self.current_epoch, self.trainer.logged_metrics.get("my_loss").item(),
            self.trainer.logged_metrics.get("valid_acc").item()))
        if self.trainer.logged_metrics.get("valid_acc").item() > self.best_acc:
            self.best_acc = self.trainer.logged_metrics.get("valid_acc").item()

    def on_train_end(self) -> None:
        self.log_record('Finished-Acc:%.3f' % self.best_acc)
        f = open('populations/after_%s.txt' % (self.file_id[4:6]), 'a+')
        f.write('%s=%.5f\n' % (self.file_id, self.best_acc))
        f.flush()
        f.close()

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime('%Y-%m-%d %H:%M:%S')
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('log/%s.txt' % (self.file_id), file_mode)
        f.write('[%s]-%s\n' % (dt, _str))
        f.flush()
        f.close()
