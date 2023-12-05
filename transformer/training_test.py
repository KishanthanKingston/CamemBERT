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