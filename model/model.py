import torch, random
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=0.2, batch_first=True)
        self.attn_mask = None
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)


    def set_attn_mask(self, attn_mask: torch.Tensor):
        self.attn_mask = attn_mask.to(dtype = bool)


    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask = attn_mask.to(device=x.device) if attn_mask is not None else None
        out, mask = self.attn(x, x, x, attn_mask=attn_mask)
        return out


    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x), self.attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Encoder(nn.Module):

    def __init__(self, n_categories: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads

        self.positional_embeddings = nn.Parameter(torch.empty(1, n_categories + 1, d_model))
        nn.init.normal_(self.positional_embeddings, std=0.02)
        
        self.blank_embeddings = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.blank_embeddings, std=0.02)
        
        self.padding_embeddings = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.padding_embeddings, std=0.02)

        self.cls_embeddings = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls_embeddings, std=0.02)

        self.encoders = [ResidualAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        self.transformer = nn.Sequential(*self.encoders)
        self.ln_post = LayerNorm(d_model)

        self.cls_head = nn.Sequential(OrderedDict([
            ("cls_dropout", nn.Dropout(p=0.5)),
            ("cls_head", nn.Linear(d_model, 2))
        ])) 

    # embeddings: [batch_size, num_categories, d_model]
    # input_mask: [batch_size, num_categories]
    def forward(self, embeddings, input_mask, target_mask, fake_embeddings, data_loader):
        # embeddings = embeddings * input_mask[:, :, None]
        # embeddings = torch.where(input_mask[:, :, None], embeddings, self.blank_embeddings)

        embeddings = torch.where(~input_mask[:, :, None] & ~target_mask[:, :, None], self.padding_embeddings, embeddings)

        if data_loader is not None:
            device = embeddings.device
            random_items = torch.empty(embeddings.shape[0] * embeddings.shape[1], embeddings.shape[-1]).to(device)
            for i in range(len(random_items)):
                random_items[i] = data_loader.get_random_item()
            random_items = random_items.reshape(embeddings.shape)

            random = torch.rand(input_mask.shape)[:, :, None].to(device)
            embeddings = torch.where(target_mask[:, :, None] & (random < 0.8), self.blank_embeddings, embeddings)
            embeddings = torch.where(target_mask[:, :, None] & (random >= 0.9), random_items, embeddings)

        else:
            embeddings = torch.where(target_mask[:, :, None], self.blank_embeddings, embeddings)

        # Add cls embeddings
        cls_embeddings_1 = self.cls_embeddings.expand(embeddings.shape[0], -1, -1)
        cls_embeddings_1 = torch.cat((cls_embeddings_1, embeddings), 1)

        # Add positional encoding
        cls_embeddings_1 = cls_embeddings_1 + self.positional_embeddings

        # # Calculate the attention mask
        # attn_mask = ~input_mask[:, None, :]
        # attn_mask = attn_mask.repeat(1, attn_mask.shape[-1], 1)
        # attn_mask = torch.repeat_interleave(attn_mask, self.n_heads, dim=0)

        # # Pass the embeddings through encoders
        # for encoder in self.encoders:
        #     encoder.set_attn_mask(attn_mask)

        out_1 = self.transformer(cls_embeddings_1)
        out_1 = self.ln_post(out_1)
        out_1 = out_1[:, 1:, :]

        out_embeddings = torch.where(target_mask[:, :, None], out_1, embeddings)
        merge_embeddings = torch.cat((out_embeddings, fake_embeddings), 0)

        # Add cls embeddings
        cls_embeddings_2 = self.cls_embeddings.expand(merge_embeddings.shape[0], -1, -1)
        cls_embeddings_2 = torch.cat((cls_embeddings_2, merge_embeddings), 1)

        # Add positional encoding
        cls_embeddings_2 = cls_embeddings_2 + self.positional_embeddings
        out_2 = self.transformer(cls_embeddings_2)
        out_2 = self.ln_post(out_2)

        out_2 = out_2[:, 0, :].squeeze()
        out_2 = self.cls_head(out_2)

        return out_1, out_2


class OutfitsTransformer(BaseModel):
    
    def __init__(self, n_categories: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        
        self.encoder = Encoder(n_categories, d_model, n_heads, n_layers, dropout)

        # self.proj = nn.Parameter(torch.randn(d_model, d_model * 4))
        # nn.init.normal_(self.proj, std=0.02)

        # self.decoder = nn.Sequential(
        #     LayerNorm(d_model * 4),
        #     QuickGELU()
        # )

        # self.out_proj = nn.Parameter(torch.randn(d_model * 4, d_model))
        # nn.init.normal_(self.out_proj, std=0.02)


    # embeddings: [batch_size, num_categories, d_model]
    # input_mask: [batch_size, num_categories]
    def forward(self, embeddings, input_mask, target_mask, fake_embeddings, data_loader):
        out = self.encoder(embeddings, input_mask, target_mask, fake_embeddings, data_loader)
        # out = out @ self.proj
        # out = self.decoder(out)
        # out = out @ self.out_proj
        return out