from torchinfo import summary


# orig imports
import logging
import os
os.environ['HYDRA_FULL_ERROR'] = '1'
from argparse import Namespace
import torch
from torchvision.utils import make_grid
import hydra
from omegaconf import DictConfig
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger as Logger
from pytorch_lightning.callbacks import ModelCheckpoint
import networks
from dataloaders import SpectrogramsDataModule
from dataloaders import ImagesDataModule
import wandb
import lmdb
from utils.helpers import extract_latent



class VQEngine(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.net = networks.VQVAE(**self.hparams.net)


model = networks.VQVAE()
batch_size = 6
# [6, 1, 128, 172]
summary(model, input_size=(3, 128, 172))