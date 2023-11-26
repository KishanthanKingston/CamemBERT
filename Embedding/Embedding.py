"""
Project : CamemBERT
Unit : Advanced Machine Learning 
MSc. Intelligent systems engineering
SORBONNE UNIVERSITÉ

--- Students ---
@SSivanesan - Shivamshan SIVANESAN
@Emirtas7 - Emir TAS
@KishanthanKingston - Kishanthan KINGSTON
"""

import numpy as np

class EmbeddingGenerator:
    def __init__(self, d_model):
        self.d_model = d_model

    def segment_embedding(self, segment_ids):
            """
        La fonction permet de créer une matrice d'embedding de segment.

        Entrée:
        - segment_ids (numpy array): Un tableau d'identifiants de segment pour chaque élément de la séquence.
        - d_model (int): La dimension de l'embedding.

        Sortie:
        - numpy array
        """
        unique_segments = np.unique(segment_ids)
        num_segments = len(unique_segments)
        
        segment_embedding_matrix = np.zeros((len(segment_ids), self.d_model))

        for i, segment_id in enumerate(unique_segments):
            mask = (segment_ids == segment_id)
            num_elements = np.sum(mask)
            
            hash_values = (hash(str(segment_id)) % 10000) / 10000.0
            embedding = np.full((num_elements, self.d_model), hash_values)
            segment_embedding_matrix[mask] = embedding

        return segment_embedding_matrix

    def positional_embedding(self, max_len):
            """
        Crée une matrice d'embedding positionnel.

        Entrée:
        - max_len (int): La longueur maximale de la séquence.
        - d_model (int): La dimension de l'embedding.

        Sortie:
        - numpy array
        """
        pos = np.arange(max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pos_embedding = np.zeros((max_len, self.d_model))

        pos_embedding[:, 0::2] = np.sin(pos * div_term)
        pos_embedding[:, 1::2] = np.cos(pos * div_term)

        return pos_embedding
