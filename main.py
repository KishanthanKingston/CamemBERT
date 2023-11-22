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
import random
# Main file

if __name__ == "__main__":
    


    pre_process = PreProcessing('fr_part_1.txt')

    input = pre_process.read_dataset ()
    masked_tokens = pre_process.dynamic_masking(pre_process.sentence_token(input[random.randint(0,100)])['input_ids'])

    print(masked_tokens)

