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
from tokenizers import SentencePieceBPETokenizer

torch.manual_seed(42) # For reproductibilty
class PreProcessing:
    def __init__(self,path:str) -> None:
        self.max_seq_len = 512
        self.path = path
        self.vocab_size = 32000
        self.sp = spm.SentencePieceProcessor()
        model_path = 'm_model.model'
        # Verify if the spm model does exist 
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The SentencePiece model file '{model_path}' does not exist.")
        self.sp.load(model_path)
        self.mask_token_id = self.sp.piece_to_id('<mask>')
        self.pad_token_id = self.sp.piece_to_id('<pad>')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sentence_piece(self) -> None:
        """
        Input
        -----
        None 

        Output
        ------
        None 

        Trains the SentencePiece Model for the custom dataset for vocab size of 3200 and with special symbols such as <s>,</s>,<mask>,<pad> and 
        creates a .model file and .vocab file
        """ 
        spm.SentencePieceTrainer.train(
           f'--input={self.path} --model_prefix=m_model --vocab_size=32000 --user_defined_symbols=<s>,</s>,<mask>,<pad>'
        )
    def sentence_token(self, txt:str,pad:bool = True) -> torch.Tensor:
        """
        Input
        -----
        txt : str
        pad : bool

        Output
        ------
        torch.Tensor

        Tokenizes a given input and also adds the padding automatically and doesn't if the pad value is set to False

        """
        sentences = txt.split('.')
        processed_text = ' '.join(f'<s> {sentence.strip()} </s>' for sentence in sentences if sentence.strip())
        
        self.tokens = torch.as_tensor(self.sp.encode_as_ids(processed_text)).to(self.device)

        self.len_token = len(self.tokens) # length of the token before padding 
        # print(self.len_token)
        # we add the padding 
        if pad:
            self.tokens = self.padding(self.tokens)

        return self.tokens,self.len_token

    def read_dataset(self, data = None) -> list[str]:
        """
        Input
        -----
        data

        Output
        ------
        list[str]

        Reads the dataset which is a txt file and returns a list of strings 
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"The file '{self.path}' does not exist.")
        
        with open(self.path,'r',encoding='utf-8') as text:
            input_data = text.readlines()

        return input_data

    def padding(self,tokens:torch.Tensor) -> torch.Tensor:
        """
        Input
        -----
        tokens : torch.Tensor

        Output
        ------
        torch.Tensor

        Does the padding to a token to ensure that all the inputs have the same length

        """
        if len(tokens) < self.max_seq_len:
            padding = torch.full((self.max_seq_len - len(tokens),), self.pad_token_id, dtype=tokens.dtype,device=tokens.device) # we create the padding tensor containing only with the pad_token_id 
            padded_tokens = torch.cat((tokens, padding)) # we then concatenate the the tokens and the padding 
        else:
            padded_tokens = tokens[:self.max_seq_len]

        return padded_tokens
    
    def dynamic_masking(self, x: torch.Tensor,len_token:int = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input
        -----
        x : torch.Tensor
        len_token : int 

        Output
        ------
        tuple[torch.Tensor, torch.Tensor]

        Dynamically masks the token by choosing 15% of the original tokens where 80% of them are masked, 10% are randomly replaced by a random token and the rest are unchanged
        """
        
        mask_label = torch.full((len_token,), False, dtype=torch.bool, device=x.device)

        # Randomly select 15% of the tokens for masking within the real token length
        # Generate a tensor with selection probabilities this assures that there will always be a True values inside the 
        # selection indices even when the len_token is small and having a probability of 0.15
        selection_ = torch.full((len_token,), 0.15, device=x.device)

        # Generate a tensor of indices sampled from multinomial distribution
        selection_indices = torch.multinomial(selection_, len_token, replacement=True).bool()
        
        mask_label[selection_indices] = True

        # Tensor to retrieve the masked tokens
        masked_tokens = x.clone()

        # Generate a random tensor for selection probabilities
        selection_probs = torch.rand((len_token,), device=x.device)
        
        # Replace 80% of the selected tokens with <MASK> token
        mask_indices = mask_label & (selection_probs < 0.8)
        # print('Mask shape: ',mask_indices.shape)
        masked_tokens[:len_token][mask_indices] = self.mask_token_id

        # Replace 10% of the selected tokens with original values
        unchanged_indices = mask_label & ~mask_indices & (selection_probs < 0.9)
        masked_tokens[:len_token][unchanged_indices] = x[:len_token][unchanged_indices]

        # Replace 10% of the selected tokens with random tokens
        random_indices = mask_label & ~mask_indices & ~unchanged_indices
        random_tokens = torch.randint(low=0, high=self.vocab_size, size=(random_indices.sum(),), device=x.device, dtype=torch.long)
        masked_tokens[:len_token][random_indices] = random_tokens

        # adding zeros to the labels that larger than len_token
        mask_label = torch.cat((mask_label, torch.full((x.shape[0]-len_token,), False, dtype=torch.bool, device=x.device)), dim=0)
        
        return masked_tokens, mask_label
        
    def create_dataloader(self,input:list[str],batch_size:int = 64,test_size:float = 0.2,shuffle:bool=True) -> tuple[DataLoader,DataLoader]: 
        """
        Input
        -----
        input : list[str]
        batch_size : int
        test_size : float
        shuffle : bool

        Output
        ------ 
        tuple[DataLoader,DataLoader]

        Divides the inputs which is a list of strings into training and testing set
        """
        
        # split the input into training and testing 
        train,test = train_test_split(input,test_size=test_size,shuffle=shuffle)

        # we use DataLoader to create the batches
        train_dataloader = DataLoader(train,batch_size=batch_size,shuffle=shuffle)
        test_dataloader = DataLoader(test,batch_size=batch_size,shuffle=shuffle)

        return train_dataloader,test_dataloader
    