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
from Embedding.SegmentEmbedding import segment_embedding
from Embedding.PositionalEmbedding import positional_embedding
from simpleRoBERTa import SimpleRoBERTa
import random
from MLM_RoBERTa import MLM_RoBERTa
import numpy as np
import torch.nn as nn
import torch
# Main file

if __name__ == "__main__":
    
    #pre_process = PreProcessing('fr_part_1.txt')
    # #pre_process.tokenized_data()

    #input = pre_process.read_dataset()

    # # pre_process.sentence_piece()
    #id = random.randint(0,100)
    # print(id)
    #data = input[id]
    # print(data)

    #masked_tokens,masked_label = pre_process.dynamic_masking(pre_process.sentence_token(data))
    # _tokens = pre_process.sentence_token(data)

    #print(masked_tokens)
    #print(masked_label)

    #masked_ = pre_process.mask_text(input[50])
    #print(masked_)

    # train_set,test_set = pre_process.create_dataloader(input)
    # train_features= next(iter(train_set))
    # # # print(train_features[0])
    # masked_tokens,masked_label = pre_process.dynamic_masking(pre_process.sentence_token(train_features[0]))
    # # print('Masked_tokens : ',masked_tokens.shape)
    # print('Masked_labels : ',masked_label.shape)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # First training of MLM Roberta
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('From main : ',device)
    roberta = MLM_RoBERTa(32000, 512, 512)
    roberta.to(device)
    
    criterion = nn.CrossEntropyLoss()
    roberta.train_mlm(criterion)
    
    torch.save(roberta.state_dict(), "MLM_RoBERTa.pth")











