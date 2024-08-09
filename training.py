import torch
import data_handling
import lightning_module
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os
import sys
from lightning.pytorch.callbacks import TQDMProgressBar


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


def training_loop() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    model = lightning_module.LightningModule()

    data_module = data_handling.DataModule()

    logger = TensorBoardLogger(save_dir=os.getcwd(), name="lightning_logs", default_hp_metric=False)

    trainer = L.Trainer(max_epochs=100, logger=logger, fast_dev_run=False, accelerator="gpu",
                        callbacks=[MyProgressBar(), EarlyStopping(monitor="valid_acc", mode="max", patience=100)])

    trainer.fit(model, data_module)
