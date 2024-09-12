# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging


import torch
import torch.nn as nn
from torch.nn import Dropout, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import numpy as np


from .encoder_decoder import unet_TSB_encoder, unet_TSB_decoder
from .Transformer_utils import AbsolutePositionalEncoder, Block, np2th

logger = logging.getLogger(__name__)




class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, hidden_size, dropout_rate):
        super(Embeddings, self).__init__()
        patch_size = (1, 1)  # patch size equals to a pixel of the feature map
        self.hybrid_model = unet_TSB_encoder(ngf=128, input_nc=1)
        # output channels of the self.hybrid_model * downsample ratio
        in_channels = self.hybrid_model.ngf * self.hybrid_model.time_downsample_ratio

        # in_channels = 128 * 2^3 
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        self.dropout = Dropout(dropout_rate)


    def forward(self, x):
        x, features, origin_len, pad_input = self.hybrid_model(x)
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        b, n, hidden = x.shape


        pos_embedding = AbsolutePositionalEncoder(hidden)
        embeddings = x + pos_embedding(x)
        embeddings = self.dropout(embeddings)

        return embeddings, features, origin_len, pad_input



class Encoder(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 transformer_num_layers, 
                 mlp_dim, 
                 num_heads, 
                 transformer_dropout_rate, 
                 transformer_attention_dropout_rate):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_size, eps=1e-6)
        for _ in range(transformer_num_layers):
            layer = Block(hidden_size, mlp_dim, num_heads, transformer_dropout_rate, transformer_attention_dropout_rate)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 transformer_num_layers,
                 mlp_dim, 
                 num_heads, 
                 transformer_dropout_rate, 
                 transformer_attention_dropout_rate):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(hidden_size, transformer_dropout_rate)
        self.encoder = Encoder(hidden_size, transformer_num_layers, mlp_dim, num_heads, transformer_dropout_rate, transformer_attention_dropout_rate)

    def forward(self, input_ids):
        embedding_output, features, origin_len, pad_input = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features, origin_len, pad_input




class TransUnet(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 transformer_num_layers, 
                 mlp_dim, 
                 num_heads, 
                 transformer_dropout_rate, 
                 transformer_attention_dropout_rate):
        super(TransUnet, self).__init__()
        self.transformer = Transformer(hidden_size, transformer_num_layers, mlp_dim, num_heads, transformer_dropout_rate, transformer_attention_dropout_rate)
        self.decoder = unet_TSB_decoder(hidden_size)

    def forward(self, input):
        x, attn_weights, features, origin_len, pad_input = self.transformer(input)  # (B, n_patch, hidden)
        output = self.decoder(x, features, origin_len, pad_input)
        return output

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)


