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


# 6689783 input dans fr_part_1.txt
if __name__ == "__main__":
    pass
    # pre_process = PreProcessing('fr_part_1.txt')

    # input = pre_process.read_dataset()
    
    # pre_process.sentence_piece()
    # id = random.randint(0,100)
    # # print(id)
    # data = input[id]
    # print(data)
    
    #masked_tokens,masked_label = pre_process.dynamic_masking(pre_process.sentence_token(data))
    # _tokens = pre_process.sentence_token("On pourra toujours parler à propos d'Averroès de \"décentrement du Sujet\" " )
    # print(_tokens)

    # First training of MLM Roberta
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # print('From main : ',device)
    # roberta = MLM_RoBERTa(vocab_size=32000, ff_dim=768, hidden_size=768)
    # roberta.to(device)

    # criterion = nn.CrossEntropyLoss()
    # roberta.train_mlm(criterion)
    # roberta.test_mlm(criterion)
    
    # ## Test to verify if the current model and the saved model are the same or not 
    # torch.save(roberta.state_dict(), "MLM_RoBERTa.pth")
    # saved_state_dict = torch.load('MLM_RoBERTa.pth')

    # Print keys and sizes from the saved state dictionary
    # print('Saved model : \n')
    # for key, value in saved_state_dict.items():
    #     print(f"Key: {key}, Size: {value.size()}")
    # print('Current model : \n')
    # # Compare with the current model's architecture
    # for name, param in roberta.named_parameters():
    #     print(f"Layer: {name}, Size: {param.size()}")

    ##  Fine tunning MLM_RoBERTa model for POS-tag

    # pos_tag = MLM_Roberta_POS_Tag('MLM_RoBERTa.pth',train_set='fr_gsd-ud-train.conllu',test_set='fr_gsd-ud-test.conllu')
    # pos_tag.to(device)

    # criterion = nn.CrossEntropyLoss()
    # pos_tag.train_pos_tag(criterion)



    ## Evaluate the MLM_roBERTa model for task specific POS-tag
    # evaluate = EvaluateRoBERTa('MLM_RoBERTa.pth', 'm_model.model',ff_dim=768,vocab_size=32000)
    # sentences = evaluate.read_conll_file('fr_gsd-ud-test.conllu')
    # print(sentences[0]['text'])
    # print(len(sentences[0]['labels']))



    # evaluate.evaluate_pt(sentences[0]['text'],sentences[0]['labels'])

    
    











