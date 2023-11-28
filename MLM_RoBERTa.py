"""
Project : CamemBERT
Unit : Advanced Machine Learning 
MSc. Intelligent systems engineering
SORBONNE UNIVERSITÉ

--- Students ---
@SSivanesan - Shivamshan SIVANESAN
@Emirtas7 - Emir TAS
@KishanthanKingston - Kishanthan KINGSTON

Code inspiré de : https://github.com/shreydan/masked-language-modeling/blob/main/model.py
"""

import torch
import torch.nn as nn
import numpy as np
from transformer.PositionalEncoding import PositionalEncoding
from simpleRoBERTa import SimpleRoBERTa

class MLM_RoBERTa(nn.Module):
    def __init__(self, vocab_size, ff_dim, output_size, hidden_size=512,  num_heads=4, num_layers=6, max_len=1000, seuil=0.5):
        
        # vocab_size = la taille de vocabulaire (Dans l'artcile, vocab_size = 32000)
        # ff_dim = la dimension de sortie
        # output_size = la taille de la sortie
        # num_heads = Nombre de tête d'attention
        # num_layers = nombre de couches de tête d'attention (RoBERTa a normalement 12 couches)
        # Seuil = seuil pour la prédiction des mots utilisés
        
        super(MLM_RoBERTa, self).__init__()

        # On utilise le SimpleRoBERTa crée
        self.roberta = SimpleRoBERTa(ff_dim, output_size, hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers, max_len=max_len)

        # On utilise une couche de sortie pour la prédiction de mots masqués
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        # Seuil pour la prédiction des mots masqués
        self.seuil = seuil

    def forward(self, x):
        # Appel au modèle RoBERTa
        roberta_output = self.roberta(x)

        # Couche de sortie pour prédire les mots masqués
        logits = self.output_layer(roberta_output)
        probabilities = self.softmax(logits)

        # C'est pour prédire les mots masqués
        mots_masques_indices = (probabilities > self.seuil).nonzero()

        # On retourn la probabilités et l'indice du mots masqués
        return probabilities, mots_masques_indices
