import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from workflow.torch import module_device


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, n_embeddings, decay=0.99, reset_threshold=0.26):
        super().__init__()
        self.eval()
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings

        self.embedding = nn.Embedding(self.n_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()

        self.weight = nn.Parameter(
            torch.ones(n_embeddings) * 1, requires_grad=False
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

    def updated_weight(self, mask):
        naive_weight = (
            self.weight * self.decay
            + (1 - self.decay) * torch.sum(mask, dim=0)
        )
        return VectorQuantizer.laplace_smoothing(naive_weight)

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # print('min norm:', self.embedding.weight.norm(dim=1).min().item())

        # TODO: prenorm? see jukebox
        # flat_input = torch.tanh(flat_input) * 4
        # flat_input = (
        #     flat_input - flat_input.mean(dim=1, keepdim=True)
        # ) / (flat_input.std(dim=1, keepdim=True) + 1e-6)
        # inputs = flat_input.view(inputs.shape)

        # self.embedding.weight.data = (
        #     self.embedding.weight.data - self.embedding.weight.data.mean(dim=1, keepdim=True)
        # ) / (self.embedding.weight.data.std(dim=1, keepdim=True) + 1e-6)

        usage = (self.weight.data >= 1e-2).sum().item()
        # if self.training and usage <= 2:
        #     below_threshold = (
        #         self.weight * self.n_embeddings <= self.reset_threshold
        #     )

        #     flat_input_subset = flat_input
        #     for embedding_index in torch.where(below_threshold)[0]:
        #         distances = torch.cdist(flat_input_subset, self.embedding.weight)
        #         batch_index = distances.max(dim=1)[1].argmax()

        #         print('reset', embedding_index.item(), batch_index.item(), inputs.shape[-3:])
        #         self.embedding.weight.data[embedding_index] = (
        #             flat_input_subset[batch_index]
        #         )
        #         self.weight.data[embedding_index] = 0.1
        #         flat_input_subset = flat_input_subset[
        #             torch.arange(len(flat_input_subset)).to(inputs) != batch_index
        #         ]
        #         if len(flat_input_subset) == 0:
        #             break

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
            weighted_current = mask.t() @ flat_input.detach()
            weighted_embedding = (
                self.embedding.weight.data * self.weight.data.unsqueeze(1)
            )

            self.weight.data = self.updated_weight(mask)

            # TODO: this will decay unused embeddings towards 0
            # Seems to be from Sonnet's tf code but could be Zalando bug, seems to work better
            self.embedding.weight.data = (
                self.decay * weighted_embedding
                + (1 - self.decay) * weighted_current
            ) / self.weight.data.unsqueeze(1)

            used_embeddings = (mask >= 1).any(dim=0)
            # self.embedding.weight.data[used_embeddings] = (
            #     self.decay * weighted_embedding[used_embeddings]
            #     + (1 - self.decay) * weighted_current[used_embeddings]
            # ) / self.weight.data.unsqueeze(1)[used_embeddings]

            # self.embedding.weight.data[~used_embeddings] *= 0.99

            # self.embedding.weight.data[
            #     ~used_embeddings & (self.weight <= 1 / self.n_embeddings)
            # ] *= 0.99

            # self.embedding.weight.data *= 0.99
            self.embedding.weight.data[used_embeddings] *= 0.99
        
        commitment_loss = (
            F.mse_loss(quantized.detach(), inputs)
            # + 0.01 * (inputs ** 2).mean()
        )

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
            usage,
            indices,
        )
