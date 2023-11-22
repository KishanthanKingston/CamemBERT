#!/usr/bin/env python3
import sentencepiece as spm
import numpy as np
import os
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast,PreTrainedTokenizer


class PreProcessing:
    def __init__(self,path:str):
        self.special_tokens = ["<s>", "</s>","<unk>"]
        self.tk_tokenizer = SentencePieceBPETokenizer()
        self.path = path

    def sentence_piece(self): 
        self.tk_tokenizer.train_from_iterator(
            'sample_txt.txt',
            vocab_size=32000,
            min_frequency=2,
            show_progress=False,
            special_tokens=self.special_tokens
        )
        try:
            self.tk_tokenizer.save("token/tokenizer.json")
            print("Token saved \n")
        except Exception as e:
            print("Error : ",e)

    def sentence_token(self, txt:str):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file = "token/tokenizer.json",
            bos_token = "<s>",
            eos_token = "</s>", 
        )
        token = tokenizer(txt)
        return token

    def read_dataset(self, data = None):
        with open(self.path,'r',encoding='utf8') as text:
            input = text.readlines()

        #myfile = Path('sample_txt.txt')
        if not os.path.isfile('sample_txt.txt'):
            #myfile.unlink()
            try:
                with open('sample_txt.txt','w') as f:
                    for i in range(50):
                        line = input[i] + '\n'
                        f.writelines(line)
                print("File sample_txt.txt created ")
            except Exception :
                print("Couldn't create txt file for sentence piece")
        return input

    def dynamic_masking(self, tokens, mask_token="<MASK>"):
        selected_tokens = np.random.choice([True, False], size=len(tokens), p=[0.15, 0.85]) # on cree un array contenat des True/False de la meme taille que le token

        for i in range(len(tokens)):
            # Only modify the selected tokens
            if selected_tokens[i]:
                # Decide what to do with the token
                action = np.random.choice(["mask", "random", "keep"], p=[0.8, 0.1, 0.1])

                if action == "mask":
                    # Replace the token with the special <MASK> token
                    tokens[i] = mask_token

                elif action == "random":
                    # Replace the token with a random token
                    tokens[i] = np.random.choice(tokens)

        return tokens

