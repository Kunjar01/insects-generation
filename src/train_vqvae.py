import os
os.environ['HYDRA_FULL_ERROR'] = '1'
from argparse import Namespace
import torch
import hydra
from omegaconf import DictConfig
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger as Logger
import networks
from dataloaders import SpectrogramsDataModule


class VQEngine(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.net = networks.VQVAE(**self.hparams.net)
        self.criterion = nn.MSELoss()

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)
    
    def configure_optimizers(self,):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer
    

    def _compute_loss(self, target, output, latent_loss):
        recon_loss = self.criterion(output, target)
        loss = recon_loss + self.hparams['latent_loss_weight'] * latent_loss
        logs = {
            'mse': recon_loss,
            'latent_loss': latent_loss,
            'loss': loss,
        }
        return loss, logs


    def training_step(self, train_batch, batch_idx):
        img, label, file_path = train_batch
        out, latent_loss = self.net(img)
        latent_loss = latent_loss.mean()


        loss, logs = self._compute_loss(target=img, output=out, latent_loss=latent_loss)
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss)
        result.log_dict(logs)
        return result



@hydra.main(config_path="configs", config_name="train_vqvae")
def main(cfg: DictConfig) -> None:
    print(cfg)

    if cfg.get('debug', False):
        logger = None
    else:
        logger = Logger(project=cfg['project_name'], name=cfg['run_name'], tags=cfg['tags'])

    train_dataloader = SpectrogramsDataModule(config=cfg['dataset'])

    engine = VQEngine(Namespace(**cfg))
    trainer = pl.Trainer(
        logger=logger,
        gpus=cfg.get('gpus', 0),
        max_epochs=cfg.get('nb_epochs', 3)
    )

    # Start training
    trainer.fit(engine, train_dataloader=train_dataloader)

if __name__ == "__main__":
    main()