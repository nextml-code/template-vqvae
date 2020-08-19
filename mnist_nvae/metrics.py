import numpy as np
import torch
import ignite


def progress_metrics():
    return  dict(
        batch_loss=ignite.metrics.RunningAverage(
            output_transform=lambda output: output['loss'],
            epoch_bound=False,
            alpha=1e-7,
        ),
    )


def train_metrics():
    return dict(
        loss=ignite.metrics.RunningAverage(
            output_transform=lambda output: output['loss'],
            epoch_bound=False,
            alpha=1e-7,
        ),
        mse=ignite.metrics.RunningAverage(
            output_transform=lambda output: (
                output['predictions'].mse(output['examples'])
            ),
            epoch_bound=False,
            alpha=1e-7,
        ),
        commitment_loss=ignite.metrics.RunningAverage(
            output_transform=lambda output: (
                torch.tensor(output['predictions'].commitment_losses)
            ),
            epoch_bound=False,
            alpha=1e-7,
        ),
        sample_loss=ignite.metrics.RunningAverage(
            output_transform=lambda output: (
                torch.tensor(output['predictions'].sample_losses)
            ),
            epoch_bound=False,
            alpha=1e-7,
        ),
        perplexity=ignite.metrics.RunningAverage(
            output_transform=lambda output: (
                torch.tensor(output['predictions'].perplexities)
            ),
            epoch_bound=False,
            alpha=1e-7,
        ),
    )


def evaluate_metrics():
    return dict(
        loss=ignite.metrics.Average(
            lambda output: output['loss']
        ),
        mse=ignite.metrics.Average(lambda output: (
            output['predictions'].mse(output['examples'])
        )),
        commitment_loss=ignite.metrics.Average(lambda output: (
            torch.tensor(output['predictions'].commitment_losses)
        )),
        sample_loss=ignite.metrics.Average(lambda output: (
            torch.tensor(output['predictions'].sample_losses)
        )),
        perplexity=ignite.metrics.Average(lambda output: (
            torch.tensor(output['predictions'].perplexities)
        )),
    )
