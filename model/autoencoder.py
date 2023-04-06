import torch
import os
import cv2
from torch.optim.lr_scheduler import OneCycleLR
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# import torch.nn as nn
# import torchvision
# from pytorch_lightning.loggers import CSVLogger
# from torch.optim.lr_scheduler import OneCycleLR
# from torchsummary import summary
import pytorch_lightning as pl
import torch.nn.functional as F
from unet import UNet
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
torch.set_default_dtype(torch.float16)
#torch.set_float32_matmul_precision("medium")

im_path = []
label = []
for pth in os.listdir("/home/turin/Desktop/lizard_island/jackson/chronological/2014/r20141102_074952_lizard_d2_081_horseshoe_circle01/081_photos"):
    im_path.append("/home/turin/Desktop/lizard_island/jackson/chronological/2014/r20141102_074952_lizard_d2_081_horseshoe_circle01/081_photos/"+pth)
    label.append(pth)
print(len(im_path),len(label))


transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float16)])#,transforms.Resize(size=(408,544), interpolation=transforms.InterpolationMode("nearest"))])
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        im_path = self.data[idx]
        im_bgr = cv2.imread(im_path)
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        im = transform(im_gray)
        label = self.labels[idx]
        return im, label

training_data = CustomDataset(im_path,label)
train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True, num_workers=16)

def create_model():
    model2 = UNet(in_channels=1,
             out_channels=1,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2)
    return model2

#Lightning Model Creation
class LitResnet(pl.LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()
        self.example_input_array = torch.rand((1, 1, 1024, 1360), dtype = torch.float16)
        self.hparams_lr = lr
        self.save_hyperparameters()
        self.model = create_model()
    def forward(self, x):
        out = self.model(x)
        return out
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred= self.forward(x)
        #print(x.shape, pred.shape)
        loss = F.mse_loss(pred, x)
        self.log("my_loss", loss, prog_bar=True)
        return loss
    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(
    #         self.parameters(),
    #         lr=self.hparams_lr,
    #         momentum=0.9,
    #         weight_decay=5e-4,
    #     )
    #     return {"optimizer":optimizer}#, "lr_scheduler": scheduler_dict}
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams_lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 1600
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

model = LitResnet(lr=0.05)
model = model.to("cuda")
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger("auto_logs",log_graph=True, name="autoencoder")
trainer = Trainer(accelerator="gpu", devices=1,
    max_epochs=3,
    logger=logger,
    callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
    log_every_n_steps= 1,
    default_root_dir="/turin/docker_log/"
)
trainer.fit(model, train_dataloader)