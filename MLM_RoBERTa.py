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
from utilis.utilis import PreProcessing
import torch.optim as optim

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
        
        self.pre_process = PreProcessing('fr_part_1.txt')

        # On utilise une couche de sortie pour la prédiction de mots masqués
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        # Seuil pour la prédiction des mots masqués
        self.seuil = seuil

        learning_rate = 0.001
        parameters = self.parameters()
        self.optimizer = optim.Adam(parameters,lr=learning_rate)

    def forward(self, x):
        # Appel au modèle RoBERTa
        roberta_output = self.roberta(x)

        # Couche de sortie pour prédire les mots masqués
        logits = self.output_layer(roberta_output)
        probabilities = self.softmax(logits)
        
        # print(logits.shape)
        # On applique le masquage juste pendant l'entraînement
        if self.training:
            # Il nous faut un booléen
            #masked_logits = torch.masked_select(logits, mask_labels.bool().unsqueeze(-1).expand_as(logits)).view(-1, vocab_size)
            #masked_labels = torch.masked_select(x, mask_labels.bool()).view(-1)
            self.masked_tokens, self.masked_labels = self.pre_process.dynamic_masking(logits)
            return probabilities, self.masked_tokens, self.masked_labels

        return probabilities
    
    def train_mlm(self, loss_function, epochs=100):
        # Fonction pour entraîner le modèle MLM
        # input_Text = Entrée contenant du texte brute (par exemple plusieurs phrases)
        # Optimizer = Adam (Comme dans l'article)
        # loss_dunction = CrossEntropy (comme dans l'artciel)
        print('Starting training.....................')
        self.train()
        
        input_Text = self.pre_process.read_dataset()[:100]

        for epoch in range(epochs):
            total_loss = 0

            for inputs in input_Text:
                self.optimizer.zero_grad()

                _, masked_logits, masked_labels = self(self.pre_process.sentence_token(inputs))

                loss = loss_function(masked_logits, masked_labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            average_loss = total_loss / len(input_Text)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')

            average_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')
