import torch
import torch.nn as nn

from base import BaseModel
from collections import OrderedDict


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.attn_mask = None
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)


    def set_attn_mask(self, attn_mask: torch.Tensor):
        self.attn_mask = attn_mask


    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask = attn_mask.to(dtype=x.dtype, device=x.device) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]


    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x), self.attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class OutfitsTransformer(BaseModel):

    def __init__(self, n_categories: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.n_heads = n_heads
        self.positional_embedding = nn.Parameter(torch.empty(n_categories, d_model))
        nn.init.normal_(self.positional_embedding, std=0.01)

        self.encoders = [ResidualAttentionBlock(d_model, n_heads) for _ in range(n_layers)]
        self.transformer = nn.Sequential(*self.encoders)


    # embeddings: [batch_size, num_categories, d_model]
    def forward(self, embeddings, input_mask):
        # Calculate the attention mask
        attn_mask = input_mask[:, :, None].float()
        attn_mask = torch.bmm(attn_mask, attn_mask.transpose(1, 2)).repeat(self.n_heads, 1, 1)

        # Add positional encoding
        embeddings = embeddings + self.positional_embedding

        # Pass the embeddings through encoders
        for encoder in self.encoders:
            encoder.set_attn_mask(attn_mask)

        return self.transformer(embeddings)