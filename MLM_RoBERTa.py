"""
Project : CamemBERT
Unit : Advanced Machine Learning 
MSc. Intelligent systems engineering
SORBONNE UNIVERSITÉ

--- Students ---
@SSivanesan - Shivamshan SIVANESAN
@Emirtas7 - Emir TAS
@KishanthanKingston - Kishanthan KINGSTON

Code inspiré de : https://github.com/shreydan/masked-language-modeling/blob/main/model.py
"""
import torch
import torch.backends.mps
import torch.nn as nn
import numpy as np
from transformer.PositionalEncoding import PositionalEncoding
from simpleRoBERTa import SimpleRoBERTa
from utilis.utilis import PreProcessing
import torch.optim as optim

class CustomLR:
    def __init__(self, d_model=512, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def __call__(self, step_num=None):
        if step_num is not None:
            self.step_num = step_num
        self.step_num += 1
        arg1 = self.step_num ** -0.5
        arg2 = self.step_num * (self.warmup_steps ** -1.5)
        return self.d_model ** -0.5 * min(arg1, arg2) 


class MLM_RoBERTa(nn.Module):
    def __init__(self, vocab_size, ff_dim, output_size, hidden_size=512,  num_heads=8, num_layers=12, max_len=1000, seuil=0.5):
        
        super(MLM_RoBERTa, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # On utilise le SimpleRoBERTa crée
        self.roberta = SimpleRoBERTa(ff_dim, output_size, hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers, max_len=max_len)
        
        #self.pre_process = PreProcessing('fr_part_1.txt')
        self.pre_process = PreProcessing('Data_1_2.txt')
        
        #self.training_data,self.testing_data = self.pre_process.create_dataloader(self.pre_process.read_dataset()[:100],shuffle=True)
        self.training_data,self.testing_data = self.pre_process.create_dataloader(self.pre_process.read_dataset()[:],shuffle=True)

        # On utilise une couche de sortie pour la prédiction de mots masqués
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.vocab_size = vocab_size
        # Seuil pour la prédiction des mots masqués
        self.seuil = seuil

        learning_rate = 1e-4 
        parameters = self.parameters()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate) #, betas=(0.9, 0.98), eps=1e-9)
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=CustomLR(d_model=hidden_size))

    def forward(self, x):
        # Appel au modèle RoBERTa
        roberta_output = self.roberta(x)

        # Couche de sortie pour prédire les mots masqués
        logits = self.output_layer(roberta_output)
        probabilities = self.softmax(logits)
        
        # Apply dynamic masking to the input
        masked_input, mask_labels = self.pre_process.dynamic_masking(x)

        # Masked_input to the device
        masked_input = masked_input.to(self.device)

        roberta_output = self.roberta(masked_input)
        # Apply output layer to get logits
        logits = self.output_layer(roberta_output)
        # Apply softmax to get probabilities
        probabilities = self.softmax(logits)
        # print(masked_input)
        """
        if self.training:
            # Select the logits and labels for the masked positions
            masked_logits = torch.masked_select(logits, mask_labels.bool().unsqueeze(-1).expand(-1, self.vocab_size))
            masked_logits = masked_logits.view(-1, self.vocab_size)

            masked_labels = torch.masked_select(x, mask_labels.bool())
            masked_labels = masked_labels.view(-1)
            # if masked_logits.isnan().all():
            #     print('Logits : ',logits.shape)
            #     print('masked_logits : ',masked_logits.shape)
            #     print('masked_labels : ',masked_labels.shape)
            #     print('Mask labels : ',mask_labels)
            return probabilities, masked_logits, masked_labels
        """
        # Select the logits and labels for the masked positions
        masked_logits = torch.masked_select(logits, mask_labels.bool().unsqueeze(-1).expand(-1, self.vocab_size))
        masked_logits = masked_logits.view(-1, self.vocab_size)

        masked_labels = torch.masked_select(x, mask_labels.bool())
        masked_labels = masked_labels.view(-1)
        # if masked_logits.isnan().all():
        #     print('Logits : ',logits.shape)
        #     print('masked_logits : ',masked_logits.shape)
        #     print('masked_labels : ',masked_labels.shape)
        #     print('Mask labels : ',mask_labels)
        return probabilities, masked_logits, masked_labels

        #return probabilities, torch.tensor([]), torch.tensor([])

    def train_mlm(self, loss_function, epochs=100):
        # Fonction pour entraîner le modèle MLM
        # input_Text = Entrée contenant du texte brute (par exemple plusieurs phrases)
        # Optimizer = Adam (Comme dans l'article)
        # loss_dunction = CrossEntropy (comme dans l'artciel)
        print(f'Starting training using {self.device}.....................')
        self.to(self.device)
        self.train()

        # input_Text = self.pre_process.read_dataset()[:100]
        input_Text = next(iter(self.training_data))
        


        for epoch in range(epochs):
            total_loss = 0

            for inputs in input_Text:
                self.optimizer.zero_grad()
                #self.scheduler.zero_grad()

                # Convert to tensor and move to GPU
                inputs = self.pre_process.sentence_token(inputs).to(self.device)  
                _,masked_logits, masked_labels = self(inputs)
                
                loss = loss_function(masked_logits, masked_labels)
                # if loss.isnan().any():
                #     print('Nan in loss')
                #     print('Inputs : ',inputs)
                #     # print('Select indices :',select)
                    
                #     # print('Masked_logits : ', masked_logits.isnan().all())
                #     # print('Masked_label : ',masked_labels.isnan().all())
                #     # print('Masked logits : ',masked_logits)
                #     # print('Masked labels : ',masked_labels)
                #     break
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2,norm_type=2)
                self.optimizer.step()
                #self.scheduler.step()
                
                total_loss += loss.item()

            average_loss = total_loss / len(input_Text)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')
            
            
    def test_mlm(self, loss_function):
        # C'est pour évaluer notre modèle
        self.eval()  

        total_correct = 0
        total_samples = 0
        total_loss = 0

        test_data = next(iter(self.testing_data))

        # C'est pour les métriques
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        with torch.no_grad():
            for inputs in test_data:
                # C'est pour convertir en tensor et déplacer sur le GPU si nécessaire
                inputs = self.pre_process.sentence_token(inputs).to(self.device)

                # Les prédictions du modèle
                probabilities, masked_logits, masked_labels = self(inputs)

                loss = loss_function(masked_logits, masked_labels)
                total_loss += loss.item()

                # C'est pour calculer les métriques de classification
                if masked_labels.numel() > 0 and probabilities.numel() > 0:
                    predictions = torch.argmax(probabilities, dim=1)
                    min_length = min(masked_labels.size(0), predictions.size(0))
                    masked_labels = masked_labels[:min_length]
                    predictions = predictions[:min_length]

                    true_positives += torch.sum((predictions == 1) & (masked_labels == 1)).item()
                    false_positives += torch.sum((predictions == 1) & (masked_labels == 0)).item()
                    false_negatives += torch.sum((predictions == 0) & (masked_labels == 1)).item()

                    correct_predictions = (predictions == masked_labels).sum().item()
                    total_correct += correct_predictions
                    total_samples += len(masked_labels)
                else:
                    print("Either predictions or masked_labels is empty.")

        if total_samples > 0:
            accuracy = total_correct / total_samples
            average_loss = total_loss / len(test_data)

            # C'est pour calculer la précision, le rappel et le F1 Score
            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

            print(f'Test Accuracy: {accuracy * 100:.2f}%')
            print(f'Average Test Loss: {average_loss:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1_score:.4f}')

            return accuracy, average_loss
        else:
            print("No samples for evaluation.")
            return 0, 0

            average_loss = total_loss / len(input_Text)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')

