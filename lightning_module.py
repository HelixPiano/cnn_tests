import lightning as L
import torch
import network
import torchmetrics as tm


class LightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = network.Network()
        self.train_acc = tm.Accuracy(task="multiclass", num_classes=29)
        self.valid_acc = tm.Accuracy(task="multiclass", num_classes=29)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        y_hat = self.model(inputs)
        self.train_acc(y_hat, target)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        loss = torch.nn.functional.cross_entropy(y_hat, target)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        y_hat = self.model(inputs)
        self.valid_acc(y_hat, target)
        self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.0001)
