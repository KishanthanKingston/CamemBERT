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

# MultiHeadAttention.py

import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8) :
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads # nombre de têtes d'attention 
        self.d_model = d_model # dimension des embeddings pour chaque token = dimension d'entrée 
        
        self.query = nn.Linear(d_model, d_model) 
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.output = nn.Linear(d_model, d_model)
    
    def split_heads(self,tensor : torch.tensor) : 
        batch_size, seq_length, d_model = tensor.size()
        tensor = tensor.view(batch_size, seq_length, self.num_heads, d_model // self.num_heads)
        tensor = tensor.transpose(1,2)
        return tensor
            
    def scaled_dot_product_attention(query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e10"))  # Masquage des scores
        
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def forward(self, query, key, value):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_model)
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        
        attention_output = attention_output.transpose(1,2).contiguous()
        attention_output = attention_output.view(attention_output.size(0), -1, self.d_model)
        
        output = self.output(attention_output)
        return output, attention_weights

        