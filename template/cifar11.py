"""
from datetime import datetime
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchvision import datasets, transforms
import lightning as L
import logging
import os
import sys
import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchmetrics as tm
import warnings


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
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
        #generated_init

    def forward(self, x):
        #generate_forward

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

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_dir, train=True, download=False)
        datasets.CIFAR10(root=self.data_dir, train=False, download=False)

    def setup(self, stage=None):
        self.data_train = datasets.CIFAR10(root=self.data_dir, train=True, transform=self.transform)
        self.data_val = datasets.CIFAR10(root=self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data_train, batch_size=self.batch_size,
                                           num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data_val, batch_size=self.batch_size,
                                           num_workers=self.num_workers, pin_memory=True, persistent_workers=True)


def training_loop() -> None:
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    model = LightningModule()

    data_module = CIFAR10DataModule()

    trainer = L.Trainer(max_epochs=1, fast_dev_run=False, accelerator="gpu", logger=False, precision="bf16-mixed",
                        enable_checkpointing=False,
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

"""


