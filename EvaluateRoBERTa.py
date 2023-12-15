"""
Project : CamemBERT
Unit : Advanced Machine Learning 
MSc. Intelligent systems engineering
SORBONNE UNIVERSITÉ

--- Students ---
@SSivanesan - Shivamshan SIVANESAN
@Emirtas7 - Emir TAS
@KishanthanKingston - Kishanthan KINGSTON

Code inspiré de site officiel HuggingFace
"""

from sklearn.metrics import accuracy_score, classification_report
import torch
from utilis.utilis import PreProcessing
from MLM_RoBERTa import MLM_RoBERTa
import sentencepiece as spm
import random

class EvaluateRoBERTa:
    def __init__(self, model_path, tokenizer_path,ff_dim, vocab_size):
        
        # model_path = le chemin vers notre modèle pré-entraîné
        # tokenizer_path = le chemin vers notre tokenizer
        # output_size = la taille de la sortie
        # ff_dim = la dimension de sortie de la couche feedforward
        # vocab_size = la taille de vocabulaire (Dans l'artcile, vocab_size = 32000)
        
        # On commence par charger notre modèle
        self.model = MLM_RoBERTa(vocab_size=vocab_size,ff_dim=ff_dim)
        # self.model.load_state_dict(torch.load(model_path))    

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)

    def evaluate_pt(self, text,true_labels):
        # La fonction qu'on utilise pour évaluer pos-tagging
        tokens = self.tokenizer.encode_as_pieces(text)
        inputs = self.tokenizer.encode_as_ids(text)
        
        # Create a mask for a portion of tokens (you can customize this)
        mask_indices = [i for i in range(len(tokens)) if random.random() < 0.15]  # Example: 15% masking
        inputs = torch.as_tensor(inputs).to(self.device)
        # Apply the mask to inputs
        inputs[mask_indices] = self.tokenizer.piece_to_id('<mask>')
        # print(inputs)
        # On n'a pas besoin de calculer les gradients
        with torch.no_grad():
            self.model.to(self.device)
            outputs = self.model(inputs,len(inputs))
            predictions = torch.argmax(outputs[1], dim=-1) # On effectue l'argmax seulement sur le dernière dimension de notre couche

        print(predictions)
        pred_labels = [self.tokenizer.decode_ids(i) for i in predictions.tolist()]
        masked_pos_tags = [true_labels[i] for i, token in enumerate(tokens) if token == '<mask>']
        print('pred_label : ',pred_labels)
        print('mask pos tag : ',masked_pos_tags)
        print(len(pred_labels))
        print(len(masked_pos_tags))
        # On calcule ici notre accuracy
        # accuracy = accuracy_score(true_labels, pred_labels)
        # # Classification report de ScikitLearn
        # classificationReport = classification_report(true_labels, pred_labels)

        # return accuracy, classificationReport

    def read_conll_file(self, file_path):
        dataset = []
        with open(file_path, 'r', encoding='utf-8') as file:
            current_sentence_text = ''
            current_sentence_labels = []
            for line in file:
                line = line.strip()
                if line.startswith('# text = '):
                    current_sentence_text = line[len('# text = '):]
                elif not line or line.startswith('#'):
                    continue
                elif line.startswith('1\t'):  # Start of a new sentence
                    if current_sentence_text and current_sentence_labels:
                        dataset.append({'text': current_sentence_text, 'labels': current_sentence_labels})
                    current_sentence_labels = [line.split('\t')[3]] 
                else:
                    current_sentence_labels.append(line.split('\t')[3])

        # Add the last sentence to the list
        if current_sentence_text and current_sentence_labels:
            dataset.append({'text': current_sentence_text, 'labels': current_sentence_labels})

        return dataset
    