import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from workflow.torch import module_device


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, n_embeddings, decay=0.99, reset_threshold=1e-4):
        super().__init__()
        self.eval()
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings

        self.embedding = nn.Embedding(self.n_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()

        self.weight = nn.Parameter(
            torch.ones(n_embeddings) * 0.0, requires_grad=False
        )
        self.decay = decay
        self.reset_threshold = reset_threshold

    @staticmethod
    def laplace_smoothing(weight, epsilon=1e-5):
        n = weight.sum()
        return (
            (weight + epsilon)
            / (n + len(weight) * epsilon)
            * n
        )

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # TODO: prenorm? see jukebox

        if self.training:
            below_threshold = (self.weight <= self.reset_threshold)

            flat_input_subset = flat_input
            for embedding_index in torch.where(below_threshold)[0]:
                distances = torch.cdist(flat_input_subset, self.embedding.weight)
                batch_index = distances[:, embedding_index].argmax()
                print('reset', embedding_index.item(), batch_index.item())
                self.embedding.weight.data[embedding_index] = (
                    flat_input_subset[batch_index]
                )
                self.weight.data[embedding_index] = 1
                flat_input_subset = flat_input_subset[
                    torch.arange(len(flat_input_subset)).to(inputs) != batch_index
                ]
                if len(flat_input_subset) == 0:
                    break

        distances = torch.cdist(flat_input, self.embedding.weight)
        indices = torch.argmin(distances, dim=1)
        
        mask = torch.zeros(indices.shape[0], self.n_embeddings).to(inputs)
        mask.scatter_(dim=1, index=indices.unsqueeze(1), value=1)

        avg_probs = torch.mean(mask, dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        quantized = (
            self.embedding(indices).view(input_shape).detach()
            + inputs
            - inputs.detach()
        )

        if self.training:
            naive_weight = mask.sum(dim=0)
            current = (
                mask.t() @ flat_input
                / (naive_weight.unsqueeze(1) + 1e-7)
            )

            average_weight = (
                self.decay * self.weight.unsqueeze(1)
            )
            current_weight = (
                (1 - self.decay) * naive_weight.unsqueeze(1)
            )
            total_weight = (average_weight + current_weight + 1e-7)

            self.embedding.weight.data = (
                current_weight * self.embedding.weight.data
                + average_weight * current
            ) / total_weight

            self.weight.data = VectorQuantizer.laplace_smoothing(
                total_weight.squeeze(1)
            )

            # self.embedding.weight.data = (
            #     self.decay * self.embedding.weight.data
            #     + (1 - self.decay) * current
            # )
            # print(self.embedding.weight.data.min(), self.embedding.weight.data.max())
        
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        
        # alternative:
        # inputs.data = quantized.data
        # alternative idea:
        # just pass on quantized and let optimizer improve it
        # separate commitment loss vs embedding loss
        # IDEA: Introduce KL loss? We want the embeddings to look like normal distribution?

        # convert quantized from BHWC -> BCHW
        return (
            quantized.permute(0, 3, 1, 2).contiguous(),
            commitment_loss,
            perplexity,
            (self.weight.data >= 1e-2).sum().item(),
            indices,
        )
