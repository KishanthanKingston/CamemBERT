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

# PositionalEncoding.py

import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len=512, d=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = self.calculate_positional_encoding(seq_len, d)
        self.dropout = nn.Dropout(0.1)

    def calculate_positional_encoding(self, seq_len, d):
        pe = torch.zeros(seq_len, d)
        n = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d, 2).float() * -(torch.log(torch.Tensor([10000.0])) / d))

        pe[:, 0::2] = torch.sin(n * div_term)
        pe[:, 1::2] = torch.cos(n * div_term)

        return pe

    def forward(self, x):
        x = x + self.encoding
        return self.dropout(x)   