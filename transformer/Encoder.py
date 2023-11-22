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

# Encoder.py

import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForward 

class OneEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(OneEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = FeedForward(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Multi-Head Attention
        attention_output, _ = self.self_attention(x, x, x, mask=mask)
        x = self.layer_norm1(x + attention_output) 
        
        # FeedForward layer
        feedforward_output = self.feedforward(x)
        x = self.layer_norm2(x + feedforward_output)
        
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([OneEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
