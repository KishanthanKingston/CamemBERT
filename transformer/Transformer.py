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

# Transformer.py

import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, num_heads, d_ff):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, mask=src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return decoder_output
