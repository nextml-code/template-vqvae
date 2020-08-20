from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from workflow.torch import ModuleCompose, module_device

from mnist_nvae import architecture, problem
from mnist_nvae.architecture import module


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = architecture.Encoder(16, levels=config['levels'])

        self.latent_channels = 20
        self.eval()
        self.decoder = architecture.DecoderNVAE(
            example_features=self.encoder(torch.zeros(
                1, 3, problem.settings.HEIGHT, problem.settings.WIDTH
            )),
            latent_channels=self.latent_channels,
            level_sizes=[
                (config['levels'] + 1 - index)
                for index in range(config['levels'])
            ]
        )

        def add_sn(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if len(m._forward_pre_hooks) == 0:
                    return torch.nn.utils.spectral_norm(m)
            else:
                return m

        self.apply(add_sn)

    def forward(self, image_batch):
        image_batch = image_batch.permute(0, 3, 1, 2).to(module_device(self))
        features = self.encoder(image_batch)
        predicted_image, commitment_losses, sample_losses, perplexities, usages = (
            self.decoder(features)
        )
        return architecture.PredictionBatch(
            predicted_image=predicted_image.permute(0, 2, 3, 1),
            commitment_losses=commitment_losses,
            sample_losses=sample_losses,
            perplexities=perplexities,
            usages=usages,
        )

    def prediction(self, features_batch: architecture.FeaturesBatch):
        return self(features_batch.image_batch)

    def generated(self, n_samples):
        predicted_image = self.decoder.generated(n_samples)
        return architecture.PredictionBatch(
            predicted_image=predicted_image.permute(0, 2, 3, 1),
        )
