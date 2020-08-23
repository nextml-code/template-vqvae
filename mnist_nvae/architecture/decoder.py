from functools import partial
import numpy as np
import torch.nn as nn
import torch.distributions as D
import torch
from torch.nn.utils import weight_norm
from workflow.torch import module_device, ModuleCompose

from mnist_nvae import tools
from mnist_nvae.architecture import module
from mnist_nvae.architecture.module import Swish


def Upsample(in_channels, out_channels):
    # TODO: this will create a checkerboard artifact?
    return ModuleCompose(
        nn.ConvTranspose2d(
            in_channels,  # channels + previous_shape[1],
            out_channels,  # channels // 2,
            kernel_size=4,
            stride=2,
            # padding=1,
            # output_padding=1,
        ),
        lambda x: x[..., 1:-1, 1:-1],
    )


class DecoderCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        expanded_channels = channels * 3
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


class AbsoluteDecoderBlock(nn.Module):
    def __init__(self, feature_shape, latent_channels, n_embeddings):
        super().__init__()
        self.feature_shape = feature_shape
        self.n_embeddings = n_embeddings
        self.latent_channels = latent_channels
        channels = feature_shape[1]
        feature_size = np.prod(feature_shape[-2:])
        
        self.quantizer = module.VectorQuantizer(
            latent_channels * feature_size,
            n_embeddings=self.n_embeddings,
        )
        self.quantized = nn.Sequential(
            nn.Conv2d(feature_shape[1], latent_channels, kernel_size=1),
            self.quantizer,
        )
        self.compute = nn.Sequential(
            nn.Conv2d(latent_channels, channels, kernel_size=1),
            DecoderCell(channels),
        )
        self.logits = nn.Parameter(
            torch.randn(self.n_embeddings) * 0.01, requires_grad=True
        )
    
    def forward(self, feature):
        quantized, commitment_loss, perplexity, usage, indices = (
            self.quantized(feature)
        )
        sample_loss = -self.distribution().log_prob(indices).mean()
        return (
            self.compute(quantized),
            commitment_loss,
            sample_loss,
            perplexity,
            usage,
        )

    def distribution(self):
        return D.Categorical(logits=self.logits)

    def generated(self, n_samples):
        indices = self.distribution().sample((n_samples,))
        return self.compute(
            self.quantizer.embedding(indices)
            .detach()
            .view(-1, *self.feature_shape[-2:], self.latent_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )


class RelativeDecoderBlock(nn.Module):
    def __init__(
        self, previous_shape, feature_shape, latent_channels, n_embeddings, upsample=True
    ):
        super().__init__()
        self.feature_shape = feature_shape
        self.n_embeddings = n_embeddings
        self.latent_channels = latent_channels
        in_channels = previous_shape[1] + feature_shape[1]
        channels = feature_shape[1]
        feature_size = np.prod(feature_shape[-2:])

        self.quantizer = module.VectorQuantizer(
            latent_channels * feature_size,
            n_embeddings=self.n_embeddings,
        )
        self.quantized = ModuleCompose(
            lambda previous, feature: (
                torch.cat([previous, feature], dim=1)
            ),
            DecoderCell(in_channels),
            nn.Conv2d(in_channels, latent_channels, kernel_size=1),
            self.quantizer,
        )
        self.compute = nn.Sequential(
            # DecoderCell(latent_channels + previous_shape[1]),
            (
                Upsample(latent_channels + previous_shape[1], channels)
                if upsample
                else nn.Conv2d(
                    latent_channels + previous_shape[1],
                    channels,
                    kernel_size=1,
                )
            ),
        )
        self.logits = nn.Sequential(
            DecoderCell(previous_shape[1]),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(previous_shape[1], self.n_embeddings, kernel_size=1),
            nn.Flatten(),
        )

    def forward(self, previous, feature):
        quantized, commitment_loss, perplexity, usage, indices = self.quantized(
            previous, feature
        )
        sample_loss = -self.distribution(previous).log_prob(indices).mean()
        return (
            self.compute(
                torch.cat([
                    quantized,
                    previous,
                ], dim=1)
            ),
            commitment_loss,
            sample_loss,
            perplexity,
            usage,
        )

    def distribution(self, previous):
        return D.Categorical(logits=self.logits(previous))

    def generated(self, previous):
        indices = self.distribution(previous).sample()
        quantized = (
            self.quantizer.embedding(indices)
            .detach()
            .view(-1, *self.feature_shape[-2:], self.latent_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return self.compute(torch.cat([quantized, previous], dim=1))


class DecoderNVAE(nn.Module):
    def __init__(self, example_features, latent_channels, n_embeddings, level_sizes):
        super().__init__()
        print('example_feature.shape:', example_features[-1].shape)
        self.latent_channels = latent_channels
        self.latent_size = example_features[-1].shape[-2:]
        self.absolute_block = AbsoluteDecoderBlock(
            example_features[-1].shape, latent_channels, n_embeddings
        )
        previous, *_ = self.absolute_block(example_features[-1])

        relative_blocks = list()
        for example_feature, level_size in zip(
            reversed(example_features), level_sizes
        ):
            relative_block_list = list()
            for group_index in range(level_size):
                print('previous.shape:', previous.shape)
                print('example_feature.shape:', example_feature.shape)
                relative_block = RelativeDecoderBlock(
                    previous.shape,
                    example_feature.shape,
                    latent_channels,
                    n_embeddings,
                    upsample=(group_index == level_size - 1),
                )
                previous, *_ = relative_block(
                    previous, example_feature
                )
                relative_block_list.append(relative_block)
            relative_blocks.append(nn.ModuleList(relative_block_list))

        self.relative_blocks = nn.ModuleList(relative_blocks)

        print('previous.shape:', previous.shape)
        self.image = ModuleCompose(
            DecoderCell(previous.shape[1]),
            nn.BatchNorm2d(previous.shape[1]),
            nn.Conv2d(previous.shape[1], 3, kernel_size=1),
            torch.sigmoid,
        )

    def forward(self, features):
        head, commitment_loss, sample_loss, perplexity, usage = (
            self.absolute_block(features[-1])
        )

        commitment_losses = [commitment_loss]
        sample_losses = [sample_loss]
        perplexities = [perplexity]
        usages = [usage]
        for relative_block_list, feature in zip(
            self.relative_blocks, reversed(features)
        ):
            for relative_block in relative_block_list:
                head, commitment_loss, sample_loss, perplexity, usage = (
                    relative_block(head, feature)
                )
                commitment_losses.append(commitment_loss)
                sample_losses.append(sample_loss)
                perplexities.append(perplexity)
                usages.append(usage)

                # if feature.shape[-1] == 32:
                #     import pdb
                #     pdb.set_trace()
                #     relative_block2 = relative_block_list[-1]
                #     head2, *_ = relative_block2(head, feature)
                #     head2 = relative_block2.generated(head)
                #     im = self.image(head2).permute(0, 2, 3, 1)
                #     from matplotlib import pyplot
                #     pyplot.imshow(np.uint8(im[6].cpu().detach().numpy())); pyplot.show()
                    
        return (
            self.image(head),
            commitment_losses,
            sample_losses,
            perplexities,
            usages,
        )

    def generated(self, n_samples):
        head = self.absolute_block.generated(n_samples)

        for relative_block_list in self.relative_blocks:
            for relative_block in relative_block_list:
                head = relative_block.generated(head)

        return self.image(head)
