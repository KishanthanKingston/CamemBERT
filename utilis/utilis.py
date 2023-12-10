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
from torchtext.transforms import SentencePieceTokenizer

class PreProcessing:
    def __init__(self,path:str) -> None:
        self.max_seq_len = 512
        self.path = path
        self.mask_token_id = None
        self.pad_token_id = None
        self.vocab_size = 32000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        self.tokens = torch.as_tensor(sp.encode_as_ids(processed_text)).to(self.device)

        self.len_token = len(self.tokens) # length of the token before padding 
        # print(self.len_token)
        # we add the padding 
        if pad:
            self.tokens = self.padding(self.tokens)
       
        return self.tokens

    def read_dataset(self, data = None) -> list[str]:
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
            padding = torch.full((self.max_seq_len - len(tokens),), self.pad_token_id, dtype=tokens.dtype,device=tokens.device) # we create the padding tensor containing only with the pad_token_id 
            padded_tokens = torch.cat((tokens, padding)) # we then concatenate the the tokens and the padding 
        else:
            padded_tokens = tokens[:self.max_seq_len]

        return padded_tokens
    
    def dynamic_masking(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        mask_label = torch.full((self.len_token,), False, dtype=torch.bool, device=x.device)

        # Randomly select 15% of the tokens for masking within the real token length
        # Generate a tensor with selection probabilities this assures that there will always be a True values inside the 
        # selection indices even when the len_token is small and having a probability of 0.15
        selection_ = torch.full((self.len_token,), 0.15, device=x.device)

        # Generate a tensor of indices sampled from multinomial distribution
        selection_indices = torch.multinomial(selection_, self.len_token, replacement=True).bool()
        
        mask_label[selection_indices] = True

        # Tensor to retrieve the masked tokens
        masked_tokens = x.clone()

        # Generate a random tensor for selection probabilities
        selection_probs = torch.rand((self.len_token,), device=x.device)
        
        # Replace 80% of the selected tokens with <MASK> token
        mask_indices = mask_label & (selection_probs < 0.8)
        masked_tokens[:self.len_token][mask_indices] = self.mask_token_id

        # Replace 10% of the selected tokens with original values
        unchanged_indices = mask_label & ~mask_indices & (selection_probs < 0.9)
        masked_tokens[:self.len_token][unchanged_indices] = x[:self.len_token][unchanged_indices]

        # Replace 10% of the selected tokens with random tokens
        random_indices = mask_label & ~mask_indices & ~unchanged_indices
        random_tokens = torch.randint(low=0, high=self.vocab_size, size=(random_indices.sum(),), device=x.device, dtype=torch.long)
        masked_tokens[:self.len_token][random_indices] = random_tokens

        # # add the padding to the tokens 
        # masked_tokens = self.padding(masked_tokens)

        # adding zeros to the labels that larger than self.len_token
        mask_label = torch.cat((mask_label, torch.full((x.shape[0]-self.len_token,), False, dtype=torch.bool, device=x.device)), dim=0)
        
        return masked_tokens, mask_label

        
    def create_dataloader(self,input,batch_size:int = 64,test_size:float = 0.2,shuffle=True):

        # split the input into training and testing 
        train,test = train_test_split(input,test_size=test_size,shuffle=shuffle)

        # we use DataLoader to create the batches
        train_dataloader = DataLoader(train,batch_size=batch_size,shuffle=shuffle)
        test_dataloader = DataLoader(test,batch_size=batch_size,shuffle=shuffle)

        return train_dataloader,test_dataloader
    
    