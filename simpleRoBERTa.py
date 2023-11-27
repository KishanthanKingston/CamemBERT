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

import torch
import torch.nn as nn
import numpy as np

class SimpleRoBERTa(nn.Module):
    # C'est une version simplifiée de l'architecture RoBERTa pour commencer
    def __init__(self,ff_dim, output_size, input_size=768, hidden_size=512,num_heads=4, num_layers=6, max_len=1000):
        super(SimpleRoBERTa, self).__init__()

        self.positional_encoding = self._get_positional_encoding(hidden_size, max_len)

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
        self.output_layer = nn.Linear(hidden_size, output_size)

    def _get_positional_encoding(self, hidden_size, max_len=1000):
        # Cette fonction permet de générer le positional encodding
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # C'ets pour calculer des termes diviseurs pour l'encodage, avec une exponentielle pour assurer une décroissance rapide
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * -(np.log(10000.0) / hidden_size))
        positional_encoding = torch.zeros((max_len, hidden_size))
        positional_encoding[:, 0::2] = torch.sin(position * div_term) # pour les colonnes paires : PE(pos, 2i)
        positional_encoding[:, 1::2] = torch.cos(position * div_term) # pour les colonnes impaires : PE(pos, 2i+1)
        return positional_encoding.unsqueeze(0)

    def forward(self, x):
        # Ajout du positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)

        # Plusieurs couches d'attention empilées avec skip connections
        for attention_layer in self.attention_layers:
            attention_output, _ = attention_layer(x, x, x)
            x = x + attention_output
            x = nn.LayerNorm(x.size()[1:]).to(x.device)(x)

        # Feedforward avec skip connection
        ff_output = self.feedforward(x)
        x = x + ff_output
        x = nn.LayerNorm(x.size()[1:]).to(x.device)(x)

        # Couche de sortie
        output = self.output_layer(x)

        return output





