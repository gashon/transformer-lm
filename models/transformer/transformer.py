import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import numpy.typing as npt

from models.transformer.layers import TransformerBlock, RMSNorm

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_heads, d_ff, num_layers, attn_pdrop=None, residual_pdrop=None):
        super(TransformerLM, self).__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.residual_pdrop = residual_pdrop
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.norm = RMSNorm(d_model, eps=1e-5)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids, weights, vocab_size, d_model):
        # Generate token embeddings + position embeddings
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long)
        position_ids = position_ids.repeat(input_ids.shape[0], 1)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)

        x = token_embeddings + position_embeddings
        
        if self.residual_pdrop is not None:
            x = F.dropout(x, p=self.residual_pdrop, inplace=False)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Normalize and project to vocabulary size
        x = self.norm(x)
        logits = self.output_projection(x)
        
        return logits
    
    def load_batch(
        dataset: npt.NDArray, batch_size: int, context_length: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a batch of data from the dataset."""

        if torch.cuda.is_available() and 'cuda' in device:
            device = device
        else:
            device = 'cpu'  # Fallback to CPU if CUDA is not available or not requested


        inputs = np.zeros((batch_size, context_length)) #, dtype=int32
        target_labels = np.zeros((batch_size, context_length)) #, dtype=int32

        l = len(dataset) - context_length
        start_idx = np.random.randint(0, l, batch_size)
        for row, idx in enumerate(start_idx):
            
            inputs[row] = dataset[idx: idx+ context_length]
            target_labels[row] = dataset[idx+ 1 : idx+ context_length + 1]

        inputs = torch.tensor(inputs, dtype=torch.long, device=device)
        target_labels = torch.tensor(target_labels, dtype=torch.long, device=device)

        return inputs, target_labels

    def load_weights(self, weights):
        # load weights into transformer blocks 
        for i, block in enumerate(self.transformer_blocks):
            block_weights = {
                "attn.q_proj.weight": weights[f"layers.{i}.attn.q_proj.weight"],
                "attn.k_proj.weight": weights[f"layers.{i}.attn.k_proj.weight"],
                "attn.v_proj.weight": weights[f"layers.{i}.attn.v_proj.weight"],
                "attn.output_proj.weight": weights[f"layers.{i}.attn.output_proj.weight"],
                "ln1.weight": weights[f"layers.{i}.ln1.weight"],
                "ln2.weight": weights[f"layers.{i}.ln2.weight"],
                "ffn.w1.weight": weights[f"layers.{i}.ffn.w1.weight"],
                "ffn.w2.weight": weights[f"layers.{i}.ffn.w2.weight"],
            }
            block.load_weights(block_weights)


        self.token_embedding.weight.data.copy_(weights['token_embeddings.weight'])
        self.position_embedding.weight.data.copy_(weights['position_embeddings.weight'])
        self.norm.gain = weights['ln_final.weight']
        self.output_projection.weight = nn.Parameter(weights['lm_head.weight'])

