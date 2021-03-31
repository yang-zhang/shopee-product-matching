from argparse import ArgumentParser
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from dataset import ShopeeDataset, get_transforms
from models import (
    ArcFaceLossAdaptiveMargin,
    Effnet_Landmark,
    RexNet20_Landmark,
    ResNest101_Landmark,
)
import torch
import torch.optim as optim
import torch.nn.functional as F


class ShpDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dftrn,
        dfval,
        train_batch_size,
        val_batch_size,
        pin_memory,
        dataloader_num_workers=8,
        imgsz=224,
    ):
        super().__init__()
        self.dftrn, self.dfval = dftrn, dfval
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.pin_memory = pin_memory
        self.imgsz = imgsz
        self.dataloader_num_workers = dataloader_num_workers
        self.num_classes = None

    def setup(self, stage=None):
        tfms_trn, tfms_val = get_transforms(self.imgsz)
        self.num_classes = self.dftrn.label_group.nunique()
        self.i2grp = sorted(self.dftrn.label_group.unique())
        self.grp2i = {v: k for k, v in enumerate(self.i2grp)}
        self.dftrn.label_group = self.dftrn.label_group.map(self.grp2i)
        self.dfval.label_group = self.dfval.label_group.map(self.grp2i)
        self.train_dataset = ShopeeDataset(self.dftrn, transform=tfms_trn)
        self.eval_dataset = ShopeeDataset(self.dfval, transform=tfms_val)

        # get adaptive margin
        tmp = np.sqrt(
            1 / np.sqrt(self.dftrn.label_group.value_counts().sort_index().values)
        )
        self.margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.pin_memory,
        )


# loss func
def criterion(logits_m, target, arc, out_dim):
    loss_m = arc(logits_m, target, out_dim)
    return loss_m


class ShpModel(pl.LightningModule):
    def __init__(self, kernel_type, enet_type, learning_rate, num_classes, margins):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.margins = margins
        if self.hparams.enet_type == "nest101":
            ModelClass = ResNest101_Landmark
        elif self.hparams.enet_type == "rex20":
            ModelClass = RexNet20_Landmark
        else:
            ModelClass = Effnet_Landmark
        self.model = ModelClass(self.hparams.enet_type, self.num_classes)
        self.arc = ArcFaceLossAdaptiveMargin(margins=self.margins, s=80)
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        # https://www.kaggle.com/boliu0/landmark-recognition-2020-third-place-submission class enet_arcface_FINAL
        x = self.model.enet(x)
        x = self.model.swish(self.model.feat(x))
        # return F.normlize(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, ys = batch
        logits = self.model(inputs)
        preds = torch.argmax(logits, -1)
        loss = self.arc(logits, ys, self.num_classes)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_accu", self.accuracy(preds, ys), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, ys = batch
        logits = self.model(inputs)
        preds = torch.argmax(logits, -1)
        loss = self.arc(logits, ys, self.num_classes)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "valid_accu",
            self.accuracy(preds, ys),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--kernel-type", type=str, required=True)
        parser.add_argument("--enet-type", type=str, required=True)
        parser.add_argument(
            "--learning_rate", type=float, help="Learning Rate", default=1e-4
        )
        return parser
