import glob
import math
import pickle
from argparse import ArgumentParser
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from einops import asnumpy, parse_shape, rearrange, reduce
from einops.layers.torch import Rearrange, Reduce
from linear_attention_transformer.linear_attention_transformer import (
    LinearAttentionTransformer,
)
from pytorch_lightning.metrics import functional
from scipy import spatial
from torch import nn
from torch.nn import functional as F

from ml.unet import UnetModel


class LinearTransformer(nn.Module):
    def __init__(self,):
        super().__init__()

        self.proj_in = nn.Conv2d(1, 128, (1, 1))
        self.proj_out = nn.Conv1d(128, 1, (1))

        self.transformer = LinearAttentionTransformer(
            dim=128, heads=4, depth=2, max_seq_len=500 * 500, one_kv_head=True,
        )

    def forward(self, x):
        x = self.proj_in(x)
        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.transformer(x)
        x = rearrange(x, "b (h w) c -> b c (h w)", h=H, w=W)
        x = self.proj_out(x)
        x = rearrange(x, "b c (h w) -> b c h w", h=H, w=W)
        return x


class TSPClassifier(pl.LightningModule):
    def __init__(
        self, in_ch=1, learning_rate=0.0002, warm_up_step=500, **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # self.model = LinearTransformer()
        self.model = UnetModel(
            in_chans=1,
            out_chans=1,
            chans=12,
            num_pool_layers=3,
            drop_prob=0.0,
            bilinear=True,
        )
        self.loss = nn.CrossEntropyLoss()

        self.metrics = {
            # "accuracy": functional.classification.accuracy,
            # "precision": functional.classification.precision,
            # "recall": functional.classification.recall,
        }

    def forward(self, x):
        # print(f'forward.shape {x.shape}')
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat.squeeze(), y)
        self.log("1_train/train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # print(y_hat.squeeze().shape, y.shape)
        loss = self.loss(y_hat.squeeze(), y)
        # metric_y_hat = y_hat > 0.5
        self.log("2_val/val_loss", loss)

        # for metric_name, metric_function in self.metrics.items():
        #     self.log(
        #         f"2_val/{metric_name}",
        #         metric_function(pred=metric_y_hat, target=y, num_classes=2),
        #     )

        if batch_idx == 0:
            for i, (x_s, y_s, y_hat_s) in enumerate(zip(x, y, y_hat)):
                adj_matrix = torch.zeros((len(y_s), len(y_s)), device=x_s.device)

                for a, b in enumerate(y_s):
                    if b == -100:
                        break
                    adj_matrix[a, b] = 1

                img = torch.dstack([x_s, adj_matrix.unsqueeze(0), y_hat_s])
                self.logger.experiment.add_image("2_val/img", img, self.global_step)
                break

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # sch = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.1)
        # sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     opt, factor=0.1, min_lr=1e-6, patience=50, verbose=True
        # )

        def lr_foo(epoch):
            if epoch < self.hparams.warm_up_step:
                # warm up lr
                lr_scale = 0.1 ** (self.hparams.warm_up_step - epoch)
            else:
                lr_scale = 0.95 ** epoch

            return lr_scale

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_foo)

        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "2_val/val_loss"}


class TSPDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str):
        self.files = glob.glob("/".join([data_dir, "*.pkl"]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as pickleFile:
            res = pickle.load(pickleFile)
            sample_adj_matrix = self.generate_adjacency_matrix(res["locations"])
            target_adj_matrix = self.generate_target_adjacency_matrix(res["solution"])
        return sample_adj_matrix.unsqueeze(0), target_adj_matrix

    def generate_adjacency_matrix(
        self, locations: List[List[float]]
    ) -> torch.FloatTensor:
        locations = np.array(locations)
        return torch.FloatTensor(spatial.distance.cdist(locations, locations))

    def generate_target_adjacency_matrix(
        self, solution_path: List[int]
    ) -> torch.LongTensor:
        # adj_matrix = np.zeros((len(solution_path), len(solution_path)))

        # for a, b in zip(
        #     solution_path + [solution_path[-1]], solution_path[1:] + [solution_path[0]]
        # ):
        #     adj_matrix[a, b] = 1
        #     adj_matrix[b, a] = 1

        return torch.LongTensor(solution_path)


class TSPDataModule(pl.LightningDataModule):
    def __init__(self, training_data_dir: str, val_data_dir: str, batch_size: int):
        super().__init__()
        self.training_data_dir = training_data_dir
        self.val_data_dir = val_data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = TSPDataset(data_dir=self.training_data_dir)
        self.val_dataset = TSPDataset(data_dir=self.val_data_dir)

    def pad_img_to_max_size(self, batch: List[torch.Tensor]) -> torch.Tensor:
        channel_first = True
        xx = [t[0] for t in batch]
        yy = [t[1] for t in batch]

        A, B = max(t.shape for t in xx)[1:]

        def padd(in_tensors, A, B, channel_first):
            tensors = []

            for tensor in in_tensors:
                if len(tensor.shape) == 3:
                    _, C, D = tensor.shape
                else:
                    C, D = tensor.shape
                    tensor = tensor.unsqueeze(0)
                if channel_first:
                    tensor = tensor.transpose(0, -1)
                tensor = F.pad(
                    tensor,
                    (
                        0,
                        0,
                        math.ceil((B - D) / 2),
                        math.floor((B - D) / 2),
                        math.ceil((A - C) / 2),
                        math.floor((A - C) / 2),
                    ),
                    value=0,
                )
                if channel_first:
                    tensor.transpose(-1, 0)
                tensors.append(tensor)
            return torch.stack(tensors)

        padded_xx = padd(xx, A, B, channel_first)
        if channel_first:
            padded_xx = padded_xx.transpose(-1, 1)

        A = max(t.shape for t in yy)

        # padded_yy = padd(yy, A, B, channel_first)

        padded_yy = torch.nn.utils.rnn.pad_sequence(
            yy, padding_value=-100, batch_first=True
        )

        return padded_xx, padded_yy

    def train_dataloader(self):
        # collate_fn=self.pad_collate
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
            collate_fn=self.pad_img_to_max_size,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            collate_fn=self.pad_img_to_max_size,
        )


def cli_main():
    pl.seed_everything(42)

    # args
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=2, type=int)
    parser.add_argument("--experiment_name", default="linformer_concorde", type=str)
    args = parser.parse_args()

    # data
    data_module = TSPDataModule(
        training_data_dir="/home/adrian/projects/travelling-salesman-problem-pytorch/train_heuristic/data/training",
        val_data_dir="/home/adrian/projects/travelling-salesman-problem-pytorch/train_heuristic/data/validation",
        batch_size=64,
    )

    # model
    model = TSPClassifier(in_ch=1, **vars(args))

    # training
    logger = pl.loggers.TestTubeLogger(
        save_dir="lightning_logs", name=args.experiment_name
    )
    logger.log_hyperparams({k: str(v) for k, v in args._get_kwargs()})

    trainer = pl.Trainer(
        logger=logger,
        gpus=args.gpus,
        distributed_backend="ddp",
        # accumulate_grad_batches=4,
        # precision=16,
    )
    trainer.fit(model, data_module)

    trainer.test(datamodule=data_module)


if __name__ == "__main__":  # pragma: no cover
    cli_main()
