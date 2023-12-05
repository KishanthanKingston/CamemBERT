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
import sentencepiece as spm
import numpy as np
import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch


class PreProcessing:
    def __init__(self,path:str) -> None:
        self.max_seq_len = 512
        self.path = path
        self.mask_token_id = None
        self.pad_token_id = None

    def sentence_piece(self) -> None: 
       spm.SentencePieceTrainer.train(
           '--input=sample_txt.txt --model_prefix=m_model --vocab_size=32000 --user_defined_symbols=<s>,</s>,<mask>,<pad>'
       )
    def sentence_token(self, txt:str,pad:bool = True) -> torch.Tensor:

        sentences = txt.split('.')
        processed_text = ' '.join(f'<s> {sentence.strip()} </s>' for sentence in sentences if sentence.strip())
        sp = spm.SentencePieceProcessor()
        sp.load('m_model.model')
        # print(sp.encode_as_pieces(processed_text))

        self.mask_token_id = sp.piece_to_id('<mask>')
        self.pad_token_id = sp.piece_to_id('<pad>')
        
        tokens = torch.as_tensor(sp.encode_as_ids(processed_text))

        self.len_token = len(tokens) # length of the token before padding 
        # we add the padding 
        if pad:
            tokens = self.padding(tokens)

        return tokens

    def read_dataset(self, data = None):
        with open(self.path,'r',encoding='utf-8') as text:
            input = text.readlines()

        if not os.path.isfile('sample_txt.txt'):
            #myfile.unlink()
            try:
                with open('sample_txt.txt','w',encoding='utf-8') as f:
                    line = [line for line in input[:int(len(input)*0.80)]]
                    f.writelines(line)
                    f.close()
                print("File sample_txt.txt created ")
            except Exception :
                print("Couldn't create txt file for sentence piece")
        return input

    def padding(self,tokens:torch.Tensor) -> torch.Tensor:
        if len(tokens) < self.max_seq_len:
            padding = torch.full((self.max_seq_len - len(tokens),), self.pad_token_id, dtype=tokens.dtype) # we create the padding tensor containing only with the pad_token_id 
            padded_tokens = torch.cat((tokens, padding)) # we then concatenate the the tokens and the padding 
        else:
            padded_tokens = tokens[:self.max_seq_len]

        return padded_tokens

    def dynamic_masking(self, logits: torch.Tensor):

        selection_mask = torch.rand(logits.shape) < 0.15
        
        # Create a tensor to store the actions for each token
        actions = torch.multinomial(torch.tensor([0.8, 0.1, 0.1]), logits.numel(), replacement=True).reshape(logits.shape) # we reshape the action tensor to the same shape as the logits 
        
        # Mask logits with the 'mask' action
        mask_indices = (actions == 0) & selection_mask
        masked_logits = logits.clone() # create a clone of the logits tensor
        masked_logits[mask_indices] = float('-inf') # we set the probability of the mask to negative infinity so that it doesn't get picked during training
        
        # Replace logits with a random token for the 'random' action
        random_indices = (actions == 1) & selection_mask
        random_logits = torch.index_select(logits.view(-1), 0, torch.randperm(logits.numel())[:random_indices.sum()])
        masked_logits[random_indices] = random_logits

        # Create masked_label for the entire sequence
        masked_labels = torch.zeros_like(logits)
        masked_labels[mask_indices] = 1 #  This defines the indices where the mask_indices are applied not the random_indices. 
        
        return masked_logits, masked_labels
    

    def create_dataloader(self,input,batch_size:int = 64,test_size:float = 0.2,shuffle=True):

        # split the input into training and testing 
        train,test = train_test_split(input,test_size=test_size,shuffle=shuffle)

        # we use DataLoader to create the batches
        train_dataloader = DataLoader(train,batch_size=batch_size,shuffle=shuffle)
        test_dataloader = DataLoader(test,batch_size=batch_size,shuffle=shuffle)

        return train_dataloader,test_dataloader
    
    