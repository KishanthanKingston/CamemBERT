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
from transformers import PreTrainedTokenizerFast
import pandas as pd 
from torch.hub import load
class PreProcessing:
    def __init__(self,path:str):
        self.max_seq_len = 512
        self.path = path
        self.mask_token_id = None

    def sentence_piece(self): 
       spm.SentencePieceTrainer.train(
           '--input=sample_txt.txt --model_prefix=m_model --vocab_size=32000 --user_defined_symbols=<s>,</s>,<mask>,<pad>'
       )
    def sentence_token(self, txt:str):
        # self.tokenizer = PreTrainedTokenizerFast(
        #     tokenizer_file = "m_model.model",
        #     mask_token = "<mask>"
        # )
        # self.mask_token_id = self.tokenizer.mask_token_id
        # print(self.mask_token_id)
        # token = self.tokenizer.encode(txt)
        # return token
        sentences = txt.split('.')
        processed_text = ' '.join(f'<s> {sentence.strip()} </s>' for sentence in sentences if sentence.strip())
        sp = spm.SentencePieceProcessor()
        sp.load('m_model.model')
        print(sp.encode_as_pieces(processed_text))

        self.mask_token_id = sp.piece_to_id('<mask>')
        self.pad_token_id = sp.piece_to_id('<pad>')
        
        tokens = sp.encode_as_ids(processed_text)
        return tokens

    def read_dataset(self, data = None):
        with open(self.path,'r',encoding='utf-8') as text:
            input = text.readlines()

        # print(len(input))
        #myfile = Path('sample_txt.txt')
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

    def dynamic_masking(self, tokens):
        selected_tokens = np.random.choice([True, False], size=len(tokens), p=[0.15, 0.85])

        for i in range(len(tokens)):
            if selected_tokens[i]:
                action = np.random.choice(["mask", "random", "keep"], p=[0.8, 0.1, 0.1])

                if action == "mask":
                    tokens[i] = self.mask_token_id

                elif action == "random":
                    tokens[i] = np.random.choice(tokens)

        if len(tokens) < self.max_seq_len:

            pad_token_id = self.pad_token_id
            # print(pad_token_id)
            tokens += [pad_token_id] * (self.max_seq_len - len(tokens))
        else:
            tokens = tokens[:self.max_seq_len]
        return tokens
    
    def decode_tokens(self,ouput):
        sp = spm.SentencePieceProcessor()
        sp.load('m_model.model')

        
