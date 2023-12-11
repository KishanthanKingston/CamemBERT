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
import torch.backends.mps
import torch.nn as nn
import numpy as np
from transformer.PositionalEncoding import PositionalEncoding
from simpleRoBERTa import SimpleRoBERTa
from utilis.utilis import PreProcessing
import torch.optim as optim

class MLM_RoBERTa(nn.Module):
    def __init__(self, vocab_size, ff_dim, output_size, hidden_size=512,  num_heads=4, num_layers=6, max_len=1000, seuil=0.5):
        
        super(MLM_RoBERTa, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # On utilise le SimpleRoBERTa crée
        self.roberta = SimpleRoBERTa(ff_dim, output_size, hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers, max_len=max_len)
        
        #self.pre_process = PreProcessing('fr_part_1.txt')
        self.pre_process = PreProcessing('Data_1_2.txt')
        
        #self.training_data,self.testing_data = self.pre_process.create_dataloader(self.pre_process.read_dataset()[:100],shuffle=True)
        self.training_data,self.testing_data = self.pre_process.create_dataloader(self.pre_process.read_dataset()[:],shuffle=True)

        # On utilise une couche de sortie pour la prédiction de mots masqués
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.vocab_size = vocab_size
        # Seuil pour la prédiction des mots masqués
        self.seuil = seuil

        learning_rate = 1e-4 
        parameters = self.parameters()
        self.optimizer = optim.Adam(parameters,lr=learning_rate)

    def forward(self, x):
        # Appel au modèle RoBERTa
        roberta_output = self.roberta(x)

        # Couche de sortie pour prédire les mots masqués
        logits = self.output_layer(roberta_output)
        probabilities = self.softmax(logits)
        
        # Apply dynamic masking to the input
        masked_input, mask_labels = self.pre_process.dynamic_masking(x)

        # Masked_input to the device
        masked_input = masked_input.to(self.device)

        roberta_output = self.roberta(masked_input)
        # Apply output layer to get logits
        logits = self.output_layer(roberta_output)
        # Apply softmax to get probabilities
        probabilities = self.softmax(logits)
        # print(masked_input)
        if self.training:
            # Select the logits and labels for the masked positions
            masked_logits = torch.masked_select(logits, mask_labels.bool().unsqueeze(-1).expand(-1, self.vocab_size))
            masked_logits = masked_logits.view(-1, self.vocab_size)

            masked_labels = torch.masked_select(x, mask_labels.bool())
            masked_labels = masked_labels.view(-1)
            # if masked_logits.isnan().all():
            #     print('Logits : ',logits.shape)
            #     print('masked_logits : ',masked_logits.shape)
            #     print('masked_labels : ',masked_labels.shape)
            #     print('Mask labels : ',mask_labels)
            return probabilities, masked_logits, masked_labels

        return probabilities

    def train_mlm(self, loss_function, epochs=100):
        # Fonction pour entraîner le modèle MLM
        # input_Text = Entrée contenant du texte brute (par exemple plusieurs phrases)
        # Optimizer = Adam (Comme dans l'article)
        # loss_dunction = CrossEntropy (comme dans l'artciel)
        print(f'Starting training using {self.device}.....................')
        self.to(self.device)
        self.train()

        # input_Text = self.pre_process.read_dataset()[:100]
        input_Text = next(iter(self.training_data))

        for epoch in range(epochs):
            total_loss = 0

            for inputs in input_Text:
                self.optimizer.zero_grad()

                # Convert to tensor and move to GPU
                inputs = self.pre_process.sentence_token(inputs).to(self.device)  
                _,masked_logits, masked_labels = self(inputs)
                
                loss = loss_function(masked_logits, masked_labels)
                # if loss.isnan().any():
                #     print('Nan in loss')
                #     print('Inputs : ',inputs)
                #     # print('Select indices :',select)
                    
                #     # print('Masked_logits : ', masked_logits.isnan().all())
                #     # print('Masked_label : ',masked_labels.isnan().all())
                #     # print('Masked logits : ',masked_logits)
                #     # print('Masked labels : ',masked_labels)
                #     break
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2,norm_type=2)
                self.optimizer.step()
                
                total_loss += loss.item()

            average_loss = total_loss / len(input_Text)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')

