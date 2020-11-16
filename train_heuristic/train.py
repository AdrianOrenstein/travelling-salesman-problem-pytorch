import glob
import pickle
from argparse import ArgumentParser
from typing import List

import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from ml.unet import UnetModel
from pytorch_lightning.metrics import functional
from scipy import spatial
from torch import nn
from torch.nn import functional as F


class TSPClassifier(pl.LightningModule):
    def __init__(self, in_ch=1, hidden_dim=24, learning_rate=1e-3, batch_size=32, num_workers=4, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = UnetModel(
            in_chans=in_ch,
            out_chans=1,
            chans=hidden_dim,
            num_pool_layers=4,
            drop_prob=0.0,
            bilinear=True,
        )

        self.loss = nn.BCEWithLogitsLoss()

        self.metrics = {
            'accuracy': functional.classification.accuracy,
            'precision': functional.classification.precision,
            'recall': functional.classification.recall,
        }

    def forward(self, x):
        # print(f'forward.shape {x.shape}')
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y)
        self.log('1_train/train_loss', loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_hat = y_hat > 0.9
        self.log('2_val/val_loss', loss)

        for metric_name, metric_function in self.metrics.items():
            self.log(f'2_val/{metric_name}',
                     metric_function(pred=y_hat, target=y, num_classes=2)
                     )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class TSPDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str):
        self.files = glob.glob("/".join([data_dir, '*.pkl']))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as pickleFile:
            res = pickle.load(pickleFile)
            sample_adj_matrix = self.generate_adjacency_matrix(
                res['locations'])
            target_adj_matrix = self.generate_target_adjacency_matrix(
                res['solution'])
        return sample_adj_matrix.unsqueeze(0), target_adj_matrix.unsqueeze(0)

    def generate_adjacency_matrix(self, locations: List[List[float]]) -> torch.FloatTensor:
        locations = np.array(locations)
        return torch.FloatTensor(spatial.distance.cdist(locations, locations))

    def generate_target_adjacency_matrix(self, solution_path: List[int]) -> torch.FloatTensor:
        adj_matrix = np.zeros((len(solution_path), len(solution_path)))

        for a, b in zip(
            solution_path + [solution_path[-1]],
            solution_path[1:] + [solution_path[0]]
        ):
            adj_matrix[a, b] = 1
            adj_matrix[b, a] = 1

        return torch.FloatTensor(adj_matrix)


class TSPDataModule(pl.LightningDataModule):
    def __init__(self, training_data_dir: str, val_data_dir: str, batch_size: int):
        super().__init__()
        self.training_data_dir = training_data_dir
        self.val_data_dir = val_data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = TSPDataset(data_dir=self.training_data_dir)
        self.val_dataset = TSPDataset(data_dir=self.val_data_dir)
        print('training dataset', len(self.train_dataset))

    def pad_collate(self, batch):
        (xx, yy) = zip(*batch)
        x_lens = [len(x) for x in xx]
        y_lens = [len(y) for y in yy]

        xx_pad = torch.nn.utils.rnn.pad_sequence(
            xx, batch_first=True, padding_value=0)
        yy_pad = torch.nn.utils.rnn.pad_sequence(
            yy, batch_first=True, padding_value=0)

        return torch.stack(xx), torch.stack(yy)

    def train_dataloader(self):
        # collate_fn=self.pad_collate
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)


def cli_main():
    pl.seed_everything(1234)

    # args
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1, type=int)
    args = parser.parse_args()

    # data
    data_module = TSPDataModule(
        training_data_dir='/home/adrian/projects/travelling-salesman-problem-pytorch/train_heuristic/data/training',
        val_data_dir='/home/adrian/projects/travelling-salesman-problem-pytorch/train_heuristic/data/validation',
        batch_size=128,
    )

    # model
    model = TSPClassifier(in_ch=1, **vars(args))

    # training
    # , limit_train_batches=200)
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=100,
        distributed_backend='ddp'
    )
    trainer.fit(model, data_module)

    trainer.test(datamodule=data_module)


if __name__ == '__main__':  # pragma: no cover
    cli_main()
