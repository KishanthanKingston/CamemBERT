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
import numpy as np
import torch
# Main file

if __name__ == "__main__":
    
    pre_process = PreProcessing('fr_part_1.txt')
    input = pre_process.read_dataset()

    # pre_process.sentence_piece()
    data = input[random.randint(0,100)]
    print(data)
    masked_tokens = pre_process.dynamic_masking(pre_process.sentence_token(data))
    # masked_tokens = pre_process.sentence_token(data)

    print(masked_tokens)
    # print(position)
    # print(type(masked_tokens))

    segment = segment_embedding(masked_tokens,512)
    position = positional_embedding(512,512)
    # print(len(position))
    roberta = SimpleRoBERTa(2048,1)
    input_tensor = torch.tensor(segment,dtype=torch.int)
    output_tensor = roberta(input_tensor)

    print(output_tensor)

