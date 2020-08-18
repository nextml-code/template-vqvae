import pdb

from functools import partial
import numpy as np
import torch.nn as nn
import torch
from torch.nn.utils import weight_norm
from workflow.torch import module_device, ModuleCompose

from mnist_nvae import tools
from mnist_nvae.architecture import module
from mnist_nvae.architecture.module import Swish


class DecoderCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        expanded_channels = channels * 6
        self.seq = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, expanded_channels, kernel_size=1),
            nn.BatchNorm2d(expanded_channels),
            module.Swish(),
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=5, padding=2, groups=expanded_channels),
            nn.BatchNorm2d(expanded_channels),
            module.Swish(),
            nn.Conv2d(expanded_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            module.SqueezeExcitation(channels),
        )

    def forward(self, x):
        return x + self.seq(x)


def VariationalBlock(feature_shape, latent_channels):
    channels = feature_shape[1]
    feature_size = np.prod(feature_shape[-2:])
    return module.VariationalBlock(
        # feature -> sample
        sample=module.VectorQuantizerSampler(
            nn.Conv2d(feature_shape[1], latent_channels, kernel_size=1),
            module.VectorQuantizer(
                latent_channels * feature_size,
                n_embeddings=512,
            ),
        ),
        # sample -> decoded_sample
        decoded_sample=ModuleCompose(
            nn.Conv2d(latent_channels, channels, kernel_size=1),
            DecoderCell(channels),
        ),
        # decoded_sample -> upsample / previous
        upsample=ModuleCompose(
            DecoderCell(channels),
            nn.ConvTranspose2d(
                channels,
                channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),  # TODO: this will create a checkerboard artifact?
        ),
    )


def RelativeVariationalBlock(previous_shape, feature_shape, latent_channels, upsample=True):
    channels = feature_shape[1]
    feature_size = np.prod(feature_shape[-2:])
    return module.RelativeVariationalBlock(
        # previous, feature -> sample
        sample=module.VectorQuantizerSampler(
            ModuleCompose(
                lambda previous, feature: (
                    torch.cat([previous, feature], dim=1)
                ),
                DecoderCell(previous_shape[1] + feature_shape[1]),
                nn.Conv2d(previous_shape[1] + feature_shape[1], latent_channels, kernel_size=1),
            ),
            module.VectorQuantizer(
                latent_channels * feature_size,
                n_embeddings=512,
            ),
        ),
        # sample -> decoded_sample
        decoded_sample=ModuleCompose(
            nn.Conv2d(latent_channels, channels, kernel_size=1),
            DecoderCell(channels),
        ),
        # decoded_sample, previous -> upsample / previous
        upsample=ModuleCompose(
            lambda decoded_sample, previous: (
                torch.cat([decoded_sample, previous], dim=1)
            ),
            DecoderCell(channels + previous_shape[1]),
            # TODO: this will create a checkerboard artifact?
            (
                nn.ConvTranspose2d(
                    channels + previous_shape[1],
                    channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
                if upsample
                else nn.Identity()
            ),
        ),
    )


class DecoderNVAE(nn.Module):
    def __init__(self, example_features, latent_channels):
        super().__init__()
        print('example_feature.shape:', example_features[-1].shape)
        self.latent_channels = latent_channels
        self.latent_size = example_features[-1].shape[-2:]
        self.variational_block = VariationalBlock(
            example_features[-1].shape, latent_channels
        )
        previous, *_ = self.variational_block(example_features[-1])

        relative_variational_blocks = list()
        for example_feature in reversed(example_features[:-1]):
            for group_index in range(2):
                print('previous.shape:', previous.shape)
                print('example_feature.shape:', example_feature.shape)
                relative_variational_block = RelativeVariationalBlock(
                    previous.shape, example_feature.shape, latent_channels, group_index == 1
                )
                previous, *_ = relative_variational_block(
                    previous, example_feature
                )
                relative_variational_blocks.append(relative_variational_block)

        self.relative_variational_blocks = nn.ModuleList(
            relative_variational_blocks
        )

        print('previous.shape:', previous.shape)
        self.image = ModuleCompose(
            DecoderCell(previous.shape[1]),
            nn.BatchNorm2d(previous.shape[1]),
            nn.Conv2d(previous.shape[1], 3, kernel_size=1),
            torch.sigmoid,
            lambda x: x * 255,
        )

    def forward(self, features):
        head, vq_loss, perplexity = self.variational_block(features[-1])

        vq_losses = [vq_loss]
        perplexities = [perplexity]
        for index, feature in enumerate(reversed(features[:-1])):
            for inner_index in range(2):
                relative_variational_block = (
                    self.relative_variational_blocks[index * 2 + inner_index]
                )
                head, vq_loss, perplexity = relative_variational_block(head, feature)
                vq_losses.append(vq_loss)
                perplexities.append(perplexity)
        
        return (
            self.image(head),
            vq_losses,
            perplexities,
        )

    def generated(self, n_samples):
        head = self.variational_block.generated((
            n_samples, self.latent_channels, *self.latent_size
        ))

        for relative_variational_block in self.relative_variational_blocks:
            head = relative_variational_block.generated(
                head,
                (n_samples, self.latent_channels, *head.shape[-2:])
            )

        return self.image(head)
