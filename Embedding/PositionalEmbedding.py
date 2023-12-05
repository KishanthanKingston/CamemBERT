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
import numpy as np

def positional_embedding(max_len, d_model):
    """
    Crée une matrice d'embedding positionnel.

    Entrée:
    - max_len (int): La longueur maximale de la séquence.
    - d_model (int): La dimension de l'embedding.

    Sortie:
    - numpy array
    """
    pos = np.arange(max_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_embedding = np.zeros((max_len, d_model))

    # Calcul de l'embedding positionnel
    pos_embedding[:, 0::2] = np.sin(pos * div_term)
    pos_embedding[:, 1::2] = np.cos(pos * div_term)

    return pos_embedding
