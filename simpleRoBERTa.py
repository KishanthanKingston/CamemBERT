"""
Project : CamemBERT
Unit : Advanced Machine Learning 
MSc. Intelligent systems engineering
SORBONNE UNIVERSITÉ

--- Students ---
@SSivanesan - Shivamshan SIVANESAN
@Emirtas7 - Emir TAS
@KishanthanKingston - Kishanthan KINGSTON
"""

import torch.nn as nn
import numpy as np
from transformer.PositionalEncoding import PositionalEncoding
import torch.nn.functional as F

class SimpleRoBERTa(nn.Module):
    # C'est une version simplifiée de l'architecture RoBERTa pour commencer
    def __init__(self, ff_dim, hidden_size=768, num_heads=12,  num_layers=12, max_len=512):

        super(SimpleRoBERTa, self).__init__()

        # Plusieurs couches d'attention, on utilise MultiheadAttention disponible sur PyTorch
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads) for _ in range(num_layers)
        ])

        # Feedforward
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_size)
        )

        # Couche de sortie
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        # Instanciation
        self.positional_encoding_instance = PositionalEncoding(d_model=hidden_size,seq_len=max_len)
        # the d_model correspond to the dimensionality of the model's hidden states which is in this case of 768

        # Initialize weights of attention layers
        for attention_layer in self.attention_layers:
            nn.init.kaiming_uniform_(attention_layer.in_proj_weight, a=nn.init.calculate_gain('relu'))
            nn.init.kaiming_uniform_(attention_layer.out_proj.weight, a=nn.init.calculate_gain('relu'))

    def forward(self, x):
        # Ajout du positional encoding
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x = x + self.positional_encoding_instance(x)
        
        # Plusieurs couches d'attention empilées avec skip connections
        for attention_layer in self.attention_layers:
            attention_output, _ = attention_layer(x, x, x)
            x = x + F.dropout(attention_output, p=0.3, training=self.training)
            # x = x + attention_output
            x = nn.LayerNorm(x.size()[1:]).to(x.device)(x)

        # Feedforward avec skip connection
        ff_output = self.feedforward(x)
        x = x + F.dropout(ff_output, p=0.3, training=self.training)
        # x = x + ff_output
        x = nn.LayerNorm(x.size()[1:]).to(x.device)(x)

        # Couche de sortie
        output = self.output_layer(x)

        return output
