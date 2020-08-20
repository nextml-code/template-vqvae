import numpy as np
import torch
from torch import nn
from workflow.torch import module_device

from mnist_nvae.architecture import module


class VariationalBlock(nn.Module):
    def __init__(self, sample, decoded_sample, upsample):
        super().__init__()
        self.sample = sample
        self.decoded_sample = decoded_sample
        self.upsample = upsample

    def forward(self, head):
        sample, commitment_loss, perplexity, usage = self.sample(head)
        upsample = self.upsample(
            self.decoded_sample(sample)
        )
        return upsample, commitment_loss, perplexity, usage

    def generated(self, shape):
        return self.upsample(
            self.decoded_sample(
                self.sample.generated(shape)
            )
        )


class RelativeVariationalBlock(nn.Module):
    def __init__(self, sample, decoded_sample, upsample):
        super().__init__()
        self.sample = sample
        self.decoded_sample = decoded_sample
        self.upsample = upsample

    def forward(self, previous, feature):
        sample, commitment_loss, perplexity, usage = self.sample(previous, feature)
        upsample = self.upsample(
            self.decoded_sample(sample),
            previous,
        )
        return upsample, commitment_loss, perplexity, usage

    def generated(self, previous, shape):
        return self.upsample(
            self.decoded_sample(
                self.sample.generated(shape)
            ),
            previous,
        )
