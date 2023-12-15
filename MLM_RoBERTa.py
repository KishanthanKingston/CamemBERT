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
import matplotlib.pyplot as plt

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
    def __init__(self, vocab_size, ff_dim, hidden_size=768, num_heads=12,  num_layers=12, max_len=512, seuil=0.5):
        
        super(MLM_RoBERTa, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # On utilise le SimpleRoBERTa crée
        self.roberta = SimpleRoBERTa(ff_dim, hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers, max_len=max_len)
        
        self.pre_process = PreProcessing('fr_part_1.txt')
        # self.pre_process = PreProcessing('Data_1_2.txt')

        #self.training_data,self.testing_data = self.pre_process.create_dataloader(self.pre_process.read_dataset()[:100],shuffle=True)
        self.training_data,self.testing_data = self.pre_process.create_dataloader(self.pre_process.read_dataset(),shuffle=True)

        # On utilise une couche de sortie pour la prédiction de mots masqués
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.vocab_size = vocab_size
        # Seuil pour la prédiction des mots masqués
        self.seuil = seuil
        learning_rate = 1e-4 

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate,betas=(0.9, 0.98), eps=1e-9) #, betas=(0.9, 0.98), eps=1e-9)
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=CustomLR(d_model=hidden_size))

    def forward(self, x,len_token):
        # Apply dynamic masking to the input
        masked_input, mask_labels = self.pre_process.dynamic_masking(x,len_token)

        # Masked_input to the device
        masked_input = masked_input.to(self.device)

        roberta_output = self.roberta(masked_input)
        # Apply output layer to get logits
        logits = self.output_layer(roberta_output)
        # Apply softmax to get probabilities
        probabilities = self.softmax(logits)
        # Select the logits and labels for the masked positions
        masked_logits = torch.masked_select(logits, mask_labels.bool().unsqueeze(-1).expand(-1, self.vocab_size))
        masked_logits = masked_logits.view(-1, self.vocab_size)

        masked_labels = torch.masked_select(x, mask_labels.bool())
        masked_labels = masked_labels.view(-1)

        return probabilities, masked_logits, masked_labels

        #return probabilities, torch.tensor([]), torch.tensor([])

    def train_mlm(self, loss_function, epochs=20):
        print(f'Starting training using {self.device}.....................')
        self.to(self.device)
        self.train()

        input_Text = next(iter(self.training_data))
        self.train_losses = []
        self.train_accuracies = []

        for epoch in range(epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0

            for inputs in input_Text:
                self.optimizer.zero_grad()

                inputs, len_token = self.pre_process.sentence_token(inputs)
                inputs = inputs.to(self.device)
                _, masked_logits, masked_labels = self(inputs, len_token)

                loss = loss_function(masked_logits, masked_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2, norm_type=2)
                self.optimizer.step()

                total_loss += loss.item()

                # Calculate accuracy
                predictions = torch.argmax(masked_logits, dim=1)
                correct_predictions = (predictions == masked_labels).sum().item()
                total_correct += correct_predictions
                total_samples += len(masked_labels)

            average_loss = total_loss / len(input_Text)
            accuracy = total_correct / total_samples
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}, Accuracy: {accuracy}')
            self.train_losses.append(average_loss)
            self.train_accuracies.append(accuracy)
            
            
    def test_mlm(self, loss_function,epochs = 20):
        self.eval()
        self.test_accuracies = []
        self.test_loss = []
        with torch.no_grad():
            for epoch in range(epochs):
                total_correct = 0
                total_samples = 0
                total_loss = 0

                test_data = next(iter(self.testing_data))

                for inputs in test_data:
                    inputs, len_token = self.pre_process.sentence_token(inputs)
                    inputs = inputs.to(self.device)
                    probabilities, masked_logits, masked_labels = self(inputs,len_token)

                    loss = loss_function(masked_logits, masked_labels)
                    total_loss += loss.item()

                    if masked_labels.numel() > 0 and probabilities.numel() > 0:
                        predictions = torch.argmax(probabilities, dim=1)
                        min_length = min(masked_labels.size(0), predictions.size(0))
                        masked_labels = masked_labels[:min_length]
                        predictions = predictions[:min_length]

                        true_positives = torch.sum((predictions == 1) & (masked_labels == 1)).item()
                        false_positives = torch.sum((predictions == 1) & (masked_labels == 0)).item()
                        false_negatives = torch.sum((predictions == 0) & (masked_labels == 1)).item()

                        correct_predictions = (predictions == masked_labels).sum().item()
                        total_correct += correct_predictions
                        total_samples += len(masked_labels)
                    else:
                        print("Either predictions or masked_labels is empty.")
                average_testing_loss = total_loss / len(test_data)
                accuracy_test = total_correct / total_samples
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_testing_loss}, Accuracy: {accuracy_test}')
                self.test_loss.append(average_testing_loss)
                self.test_accuracies.append(accuracy_test)


    def plot_train_metrics(self,epochs):
        plt.plot(range(1, epochs + 1), self.train_losses, label="Training Loss", color="blue")
        plt.title("Training Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.show()

        plt.plot(range(1, epochs + 1), self.train_accuracies, label="Training Accuracy", color="orange")
        plt.title("Training Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.show()
        
    def plot_test_metrics(self,epochs):
        plt.plot(range(1, epochs + 1), self.test_accuracies, label="Test Accuracy", color="green")
        plt.title("Test Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.show()
