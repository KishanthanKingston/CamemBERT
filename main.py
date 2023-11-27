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

from utilis.temp import PreProcessing
# from Embedding.SegmentEmbedding import segment_embedding
# from Embedding.PositionalEmbedding import positional_embedding
from simpleRoBERTa import SimpleRoBERTa
import random
import numpy as np

# Main file

if __name__ == "__main__":
    
    pre_process = PreProcessing('fr_part_1.txt')
    input = pre_process.read_dataset()

    # pre_process.sentence_piece()
    data = input[random.randint(0,100)]
    print(data)
    masked_tokens = pre_process.dynamic_masking(pre_process.sentence_token(data))
    #masked_tokens = pre_process.sentence_token(data)

    print(masked_tokens)
    # segment = segment_embedding(masked_tokens,512)

    # position = positional_embedding(len(segment),512)

    # print(position)
    roberta = SimpleRoBERTa(2048,10)
    input_tensor = masked_tokens
    output_tensor = roberta(input_tensor)

