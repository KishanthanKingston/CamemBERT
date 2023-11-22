import numpy as np

def segment_embedding(segment_ids, d_model):
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
    
    segment_embedding_matrix = np.zeros((len(segment_ids), d_model))

    # Création de l'embedding de segment
    for i, segment_id in enumerate(unique_segments):
        mask = (segment_ids == segment_id)
        num_elements = np.sum(mask)
        
        # Utilisation d'une fonction de hachage pour créer des embeddings de segments différents
        hash_values = (hash(str(segment_id)) % 10000) / 10000.0
        embedding = np.full((num_elements, d_model), hash_values)
        segment_embedding_matrix[mask] = embedding

    return segment_embedding_matrix

