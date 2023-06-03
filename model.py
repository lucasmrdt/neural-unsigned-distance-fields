import pytorch_lightning as pl
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SequentialWithIntermediates(nn.Sequential):
    class SaveHere(nn.Identity):
        pass

    def forward(self, x):
        intermediates = []
        for module in self:
            x = module(x)
            if isinstance(module, SequentialWithIntermediates.SaveHere):
                intermediates.append(x)
        return x, intermediates


class NNOriginal(nn.Module):
    def __init__(self, hidden_dim=256, d=2, displacment_factor=0.0722):
        super().__init__()
        assert d in [2, 3], 'only 2D and 3D are supported'
        MaxPool = nn.MaxPool3d if d == 3 else nn.MaxPool2d
        Conv = nn.Conv3d if d == 3 else nn.Conv2d
        BatchNorm = nn.BatchNorm3d if d == 3 else nn.BatchNorm2d

        self.d = d

        dirs = th.cat([th.eye(self.d), th.zeros(
            (1, self.d)), -th.eye(self.d)], dim=0)
        self.displacments = displacment_factor * dirs

        self.encoder = SequentialWithIntermediates(
            SequentialWithIntermediates.SaveHere(),

            # conv_in
            Conv(1, 16, 3, padding=1, padding_mode='zeros'), nn.ReLU(),
            BatchNorm(16),
            SequentialWithIntermediates.SaveHere(),
            MaxPool(2),  # 128

            # conv_0
            Conv(16, 32, 3, padding=1, padding_mode='zeros'), nn.ReLU(),
            Conv(32, 32, 3, padding=1, padding_mode='zeros'), nn.ReLU(),
            BatchNorm(32),
            SequentialWithIntermediates.SaveHere(),
            MaxPool(2),  # 64

            # conv_1
            Conv(32, 64, 3, padding=1, padding_mode='zeros'), nn.ReLU(),
            Conv(64, 64, 3, padding=1, padding_mode='zeros'), nn.ReLU(),
            BatchNorm(64),
            SequentialWithIntermediates.SaveHere(),
            MaxPool(2),  # 32

            # conv_2
            Conv(64, 128, 3, padding=1, padding_mode='zeros'), nn.ReLU(),
            Conv(128, 128, 3, padding=1, padding_mode='zeros'), nn.ReLU(),
            BatchNorm(128),
            SequentialWithIntermediates.SaveHere(),
            MaxPool(2),  # 16

            # conv_3
            Conv(128, 128, 3, padding=1, padding_mode='zeros'), nn.ReLU(),
            Conv(128, 128, 3, padding=1, padding_mode='zeros'), nn.ReLU(),
            BatchNorm(128),
            SequentialWithIntermediates.SaveHere(),
            MaxPool(2),  # 8

            # conv_4
            Conv(128, 128, 3, padding=1, padding_mode='zeros'), nn.ReLU(),
            Conv(128, 128, 3, padding=1, padding_mode='zeros'), nn.ReLU(),
            BatchNorm(128),
            SequentialWithIntermediates.SaveHere(),
        )

        feature_size = (1 + 16 + 32 + 64 + 128 + 128 + 128) * len(dirs) + d
        self.decoder = nn.Sequential(
            nn.Conv1d(feature_size, hidden_dim * 2, 1), nn.ReLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1), nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1), nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 1),
        )

    def encode(self, x):
        x = x.unsqueeze(1)
        features = self.encoder(x)[1]
        assert len(features) == 7
        return features

    # fmt: off
    def decode(self, p, features):
        """
        p (B, N, dim)
        features (B, C, 256, 256, 256)
        """
        p_features = p.transpose(1, -1)  # (B, dim, sample_num)

        displacments = self.displacments.to(p.device)  # (#displacments, dim)

        # prepare for grid_sample
        p = p.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N, dim)
        displacments = displacments.unsqueeze(1).unsqueeze(1)  # (#displacments, 1, 1, dim)

        # augment p with displacments
        p = p + displacments

        p = p.transpose(1, 2)  # (B, 1, #displacments, N, dim)

        if self.d == 2:
            p = p.squeeze(1)  # (B, #displacments, N, dim)

        features = [F.grid_sample(f, p, padding_mode='border', align_corners=True) for f in features]
        features = th.cat(features, dim=1)  # (B, feature_size, 1, #displacments, N)

        if self.d == 3:
            features = features.squeeze(2)  # (B, feature_size, #displacments, N)

        shape = features.shape
        features = th.reshape(features, (shape[0], shape[1] * shape[2], shape[3]))  # (B, feature_size * #displacments, N)

        features = th.cat([features, p_features], dim=1)  # (B, feature_size * #displacments + dim, N)

        out = self.decoder(features).squeeze(1) # (B, N)

        return out
    # fmt: on

    def forward(self, p, point_cloud):
        return self.decode(p, self.encode(point_cloud))


class NDF(pl.LightningModule):
    def __init__(self, max_dist=.1, loss='l1', lr=1e-3, clamp=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = NNOriginal()
        if loss == 'l1':
            self.loss = nn.L1Loss(reduction='none')
        elif loss == 'l2':
            self.loss = nn.MSELoss(reduction='none')
        self.learning_rate = lr
        self.clamp = clamp
        self.max_dist = max_dist

    def forward(self, p, point_cloud):
        return self.model(p, point_cloud)

    def step(self, batch, batch_idx):
        p, vp, dist = batch
        pred = self(p, vp)
        if self.clamp:
            pred = th.clamp(pred, max=self.max_dist)
            dist = th.clamp(dist, max=self.max_dist)
        loss = self.loss(pred, dist).mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(), lr=self.learning_rate)
