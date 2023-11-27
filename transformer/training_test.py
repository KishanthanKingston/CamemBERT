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

# training_test.py

"""
from Transformer import Transformer

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from transformers import CamembertTokenizer
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, source_sentences, target_sentences, tokenizer):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        src_tokens = self.tokenizer.encode_plus(self.source_sentences[idx], padding='max_length', truncation=True, return_tensors='pt')
        tgt_tokens = self.tokenizer.encode_plus(self.target_sentences[idx], padding='max_length', truncation=True, return_tensors='pt')
        return src_tokens['input_ids'], tgt_tokens['input_ids']  # Utilisez les 'input_ids' ici


# Charger le tokeniseur CamemBERT
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

source_data = [
    "Je vais à la bibliothèque.",
    "Où est le restaurant ?",
    "Comment ça va ?"
]

target_data = [
    "I am going to the library.",
    "Where is the restaurant?",
    "How are you?"
]


# Définir votre dataset et dataloader
dataset = CustomDataset(source_data, target_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instancier votre modèle
model = Transformer(num_encoder_layers=4, num_decoder_layers=4)  # Modifier ces valeurs selon votre modèle

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement du modèle
num_epochs = 5  # Choisir le nombre d'époques d'entraînement

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for src_tokens, tgt_tokens in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        optimizer.zero_grad()
        
        src_tokens = src_tokens.squeeze(1)
        tgt_tokens = tgt_tokens.squeeze(1)
        
        # Forward pass
        output = model(src_tokens, tgt_tokens)
        output = output.view(-1, output.size(-1))
        loss = criterion(output, tgt_tokens.view(-1))
        # Backpropagation et mise à jour des poids
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Transformer import Transformer

class DummyDataset(Dataset):
    def __init__(self, num_samples=100, sequence_length=10):
        self.num_samples = num_samples
        self.sequence_length = sequence_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        src = torch.rand(512)  
        tgt = torch.rand(512) 
        return src, tgt
    
def train(model, dataloader, loss_function, optimizer):
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        optimizer.zero_grad()

        output = model(src, tgt)
        output = output.view(-1, output.shape[-1])
        tgt = tgt.view(-1, tgt.shape[-1])

        loss = loss_function(output, tgt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

dataloader = DataLoader(DummyDataset(), batch_size=32)
transformer = Transformer(num_encoder_layers=12, num_decoder_layers=12)
optimizer = torch.optim.Adam(transformer.parameters())
loss_function = nn.CrossEntropyLoss()

for epoch in range(100):  # nombre d'époques
    loss = train(transformer, dataloader, loss_function, optimizer)
    print(f"Epoch: {epoch}, Loss: {loss}")