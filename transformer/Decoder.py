"""
Project : CamemBERT
Unit : Advanced Machine Learning 
MSc. Intelligent systems engineering
SORBONNE UNIVERSITÉ

--- Students ---
@KishanthanKingston - Kishanthan KINGSTON
@SSivanesan - Shivamshan SIVANESAN
@Emirtas7 - Emir TAS
"""

# Decoder.py

import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForward 


class OneDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(OneDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = FeedForward(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Auto-attention
        self_attention_output, _ = self.self_attention(x, x, x, mask=tgt_mask)
        x = self.layer_norm1(x + self_attention_output)
        
        # Attention sur l'encodeur
        encoder_attention_output, _ = self.encoder_attention(x, encoder_output, encoder_output, mask=src_mask)
        x = self.layer_norm2(x + encoder_attention_output)
        
        # FeedForward
        feedforward_output = self.feedforward(x)
        x = self.layer_norm3(x + feedforward_output)
        
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([OneDecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return x
