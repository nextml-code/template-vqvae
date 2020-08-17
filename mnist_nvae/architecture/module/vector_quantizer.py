# Prenorm: https://github.com/openai/jukebox/blob/master/jukebox/vqvae/bottleneck.py
# 

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from workflow.torch import module_device


class VectorQuantizer(nn.Module):
    """
    Input any tensor to be quantized. Last dimension will be used as space in
    which to quantize. All other dimensions will be flattened and will be seen
    as different examples to quantize.
    The output tensor will have the same shape as the input.
    For example a tensor with shape [16, 32, 32, 64] will be reshaped into
    [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
    independently.
    Args:
        embedding_dim: integer representing the dimensionality of the tensors in the
            quantized space. Inputs to the modules must be in this format as well.
        n_embeddings: integer, the number of vectors in the quantized space.
        decay: float, decay for the moving averages.
        epsilon: small float constant to avoid numerical instability.
    """
    
    def __init__(self, embedding_dim, n_embeddings, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings

        self.embedding = nn.Embedding(self.n_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()

        self.register_buffer('_ema_cluster_size', torch.zeros(n_embeddings))
        # self._ema_cluster_size = nn.Parameter(torch.zeros(n_embeddings))
        self.ema_w = nn.Parameter(torch.Tensor(n_embeddings, embedding_dim))
        self.ema_w.data.normal_()

        self.decay = decay
        self.epsilon = epsilon

    def forward(self, inputs, compute_distances_if_possible=True, record_codebook_stats=False):
        """
        Connects the module to some inputs.

        Args:
            inputs: Tensor, final dimension must be equal to embedding_dim. All other
                leading dimensions will be flattened and treated as a large batch.
        
        Returns:
            quantize: Tensor containing the quantized version of the input.
            loss: Tensor containing the loss to optimize.
        """
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # TODO: prenorm?

        # TODO: use torch.cdist instead
        # Calculate distances
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True) 
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.n_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        # TODO: unclear why they are doing this ugly stuff?
        # Checked, seems to give same result as
        # quantized = self.embedding(encoding_indices)
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = (
                self._ema_cluster_size * self.decay
                + (1 - self.decay) * torch.sum(encodings, 0)
            )
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size.data = (
                (self._ema_cluster_size + self.epsilon)
                / (n + self.n_embeddings * self.epsilon) * n
            )
            
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)
            
            self.embedding.weight = nn.Parameter(self.ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        # loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return (
            quantized.permute(0, 3, 1, 2).contiguous(),
            e_latent_loss,
            # perplexity,
            # encodings,
        )


class VectorQuantizerSampler(nn.Module):
    def __init__(self, block, vector_quantizer):
        super().__init__()
        self.block = block
        self.vector_quantizer = vector_quantizer

    def forward(self, x):
        return self.vector_quantizer(self.block(x))

    def generated(self, shape):
        return self.vector_quantizer.embedding(
            torch.randint(self.vector_quantizer.n_embeddings, shape[:1])
            .to(module_device(self))
            # torch.LongTensor([0]).repeat(shape[0])
        ).view(*shape)
