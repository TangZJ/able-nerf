from turtle import color
from matplotlib.pyplot import get
import torch
from pdb import set_trace as st
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class AttentionEncoder(LightningModule):
    def __init__(self, input_dim=256, num_heads=8, ff_ratio=3, dropout_p=0.0):
        super().__init__()

        ff_dim = input_dim * ff_ratio
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout_p, batch_first=True)
        self.ff_layer = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(ff_dim, input_dim),
        )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, kv):
        attn_out, weights = self.attn(x, kv, kv)
        x = self.norm1(x + attn_out)

        ff_out = self.ff_layer(x)
        ff_out = self.dropout(ff_out)
        x = self.norm2(ff_out + x)

        return x, weights
        

class DecoderBlock(LightningModule):
    def __init__ (self, input_dim=256, num_heads=4, ff_ratio=3, dropout_p=0.0):
        super().__init__()

        ff_dim = input_dim * ff_ratio
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout_p, batch_first=True)
        self.ff_layer = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(ff_dim, input_dim),
        )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_p)


    def forward(self, x, attn_masks):


        attn_out, weights = self.attn(x,x,x, attn_mask=attn_masks)
        #attn_out = self.dropout(attn_out)
        x = self.norm1(x + attn_out) # add then norm

        ff_out = self.ff_layer(x)
        ff_out = self.dropout(ff_out)
        x = self.norm2(ff_out + x)

        return x, weights

 
    
class TransformerClassDecoderV2(LightningModule):
    def __init__(self,
                 layers=4, dim=256, ff_ratio=2, dropout=0.0):
        super(TransformerClassDecoderV2, self).__init__()
        self.num_heads = dim // 64
        self.layers = nn.ModuleList(
            [DecoderBlock(input_dim=dim, num_heads=self.num_heads, ff_ratio=ff_ratio, dropout_p=dropout) \
                for _ in range(layers)])

    def forward(self, tokens, attn_masks):
        batch = tokens.shape[0]        
        attn_masks = attn_masks.expand(batch * self.num_heads , -1 , -1).bool()

        for layer in self.layers:
            tokens, weights = layer(tokens, attn_masks)

        return tokens,weights

