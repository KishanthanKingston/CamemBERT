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
https://huggingface.co/Jean-Baptiste/camembert-ner
"""

from transformers import RobertaTokenizer
from sklearn.metrics import accuracy_score, classification_report
import torch
from utilis.utilis import PreProcessing

class EvaluateRoBERTa:
    def __init__(self, model_path, tokenizer_path, output_size, ff_dim, vocab_size):
        
        # model_path = le chemin vers notre modèle pré-entraîné
        # tokenizer_path = le chemin vers notre tokenizer
        # output_size = la taille de la sortie
        # ff_dim = la dimension de sortie de la couche feedforward
        # vocab_size = la taille de vocabulaire (Dans l'artcile, vocab_size = 32000)
        
        # On commence par charger notre modèle
        self.model = MLM_RoBERTa(output_size, ff_dim, vocab_size)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # On utilise le tokenizer de HuggingFace. On peut aussi utiliser notre tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path) 
        # Sur le site de HuggingFace, ils ont tokenizer = RobertaTokenizer.from_pretrained('camembert-base')

        self.pre_process = PreProcessing('fr_part_1.txt')

    def evaluate_pt(self, text, true_labels):
        # La fonction qu'on utilise pour évaluer pos-tagging
        tokens = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(text)))
        inputs = self.tokenizer.encode(text, return_tensors="pt") # pt pour PyTorch

        # On n'a pas besoin de calculer les gradients
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = torch.argmax(outputs, dim=-1) # On effectue l'argmax seulement sur le dernière dimension de notre couche

        pred_labels = [self.tokenizer.convert_ids_to_tokens(i) for i in predictions.tolist()[0]]

        # On calcule ici notre accuracy
        accuracy = accuracy_score(true_labels, pred_labels)
        # Classification report de ScikitLearn
        classificationReport = classification_report(true_labels, pred_labels)

        return accuracy, classificationReport

    def evaluate_dp(self, text, true_labels):
        # La fonction qu'on utilise pour évaluer dependency parsing
        tokens = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(text)))
        inputs = self.tokenizer.encode(text, return_tensors="pt") # pt pour PyTorch

        # On n'a pas besoin de calculer les gradients
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = torch.argmax(outputs, dim=-1) # On effectue l'argmax seulement sur le dernière dimension de notre couche

        pred_labels = predictions.tolist()[0]

        # Classification report de ScikitLearn
        classificationReport = classification_report(true_labels, pred_labels)

        return classificationReport

    def evaluate_ner(self, text, true_labels):
        # La fonction qu'on utilise pour évaluer named entity recognition
        tokens = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(text)))
        inputs = self.tokenizer.encode(text, return_tensors="pt") # pt pour PyTorch

        # On n'a pas besoin de calculer les gradients
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = torch.argmax(outputs, dim=-1) # On effectue l'argmax seulement sur le dernière dimension de notre couche

        pred_labels = predictions.tolist()[0]

        # Classification report de ScikitLearn
        classificationReport = classification_report(true_labels, pred_labels)

        return classificationReport


