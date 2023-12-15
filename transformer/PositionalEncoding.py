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
import math
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len=512, d_model=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = self.calculate_positional_encoding(seq_len, d_model)
        self.dropout = nn.Dropout(0.2)
        
        if torch.cuda.is_available():
            self.encoding = self.encoding.cuda()

    def calculate_positional_encoding(self, seq_len, d_model):
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # Use broadcasting to calculate the positional encoding
        encoding = torch.zeros(seq_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        return encoding

    def forward(self, x):
        if x.dim() == 1:
            seq_len = x.size(0)
            x = x.unsqueeze(1).to(self.encoding.device) 
        else:
            seq_len, _ = x.size()

        # print(f"x size: {x.size()}")
        # print(f"positional encoding size before: {self.encoding.size()}")
        encoding = self.calculate_positional_encoding(seq_len, self.encoding.size(1)).detach()
        encoding = encoding.to(x.device)  # Move to the same device as x
        # print(f"positional encoding size after: {encoding.size()}")
        x = x + encoding
        return self.dropout(x.squeeze(1))