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

from utilis.utilis import PreProcessing
from simpleRoBERTa import SimpleRoBERTa
import random
from MLM_RoBERTa import MLM_RoBERTa
import numpy as np
import torch.nn as nn
import torch
from EvaluateRoBERTa import EvaluateRoBERTa
from MLM_Roberta_POS_Tag import MLM_Roberta_POS_Tag
# Main file

if __name__ == "__main__":
    ## First training of MLM Roberta
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roberta = MLM_RoBERTa(vocab_size=32000, ff_dim=768, hidden_size=768)
    roberta.to(device)

    criterion = nn.CrossEntropyLoss()
    roberta.train_mlm(criterion)
    # roberta.test_mlm(criterion)
    # torch.save(roberta.state_dict(), "MLM_RoBERTa.pth")

    ## Test to verify if the current model and the saved model are the same or not 
    # saved_state_dict = torch.load('MLM_RoBERTa.pth')

    # Print keys and sizes from the saved state dictionary
    # print('Saved model : \n')
    # for key, value in saved_state_dict.items():
    #     print(f"Key: {key}, Size: {value.size()}")
    # print('Current model : \n')
    # # Compare with the current model's architecture
    # for name, param in roberta.named_parameters():
    #     print(f"Layer: {name}, Size: {param.size()}")


    
    











