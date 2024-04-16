import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import numpy.typing as npt
from typing import IO, BinaryIO, Iterable, Optional, Type
import os

from models.transformer.layers import TransformerBlock, RMSNorm


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float | None = None,
        residual_pdrop: float | None = None,
    ):
        super(TransformerLM, self).__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.residual_pdrop = residual_pdrop

        # Token and position embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(residual_pdrop)

        # Output layers
        self.ln_final = RMSNorm(d_model, eps=1e-5)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        # Generate token embeddings + position embeddings
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long)

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids).unsqueeze(0)

        x = token_embeddings + position_embeddings

        x = self.dropout(x)

        for block in self.layers:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits

    @staticmethod
    def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes],
    ):
        checkpoint = {
            "optimizer_state_dict": optimizer.state_dict(),
            "model_state_dict": model.state_dict(),
            "iteration": iteration,
        }
        torch.save(checkpoint, out)

    @staticmethod
    def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        loaded_checkpoint = torch.load(src)
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
        model.load_state_dict(loaded_checkpoint["model_state_dict"])

        return loaded_checkpoint["iteration"]

    @staticmethod
    def load_batch(
        dataset: npt.NDArray, batch_size: int, context_length: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a batch of data from the dataset."""

        if torch.cuda.is_available() and "cuda" in device:
            device = device
        else:
            device = "cpu"

        inputs = np.zeros((batch_size, context_length))
        target_labels = np.zeros((batch_size, context_length))

        l = len(dataset) - context_length
        start_idx = np.random.randint(0, l, batch_size)
        for row, idx in enumerate(start_idx):

            inputs[row] = dataset[idx : idx + context_length]
            target_labels[row] = dataset[idx + 1 : idx + context_length + 1]

        inputs = torch.tensor(inputs, dtype=torch.long, device=device)
        target_labels = torch.tensor(target_labels, dtype=torch.long, device=device)

        return inputs, target_labels
