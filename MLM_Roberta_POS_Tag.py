import torch
import torch.nn as nn
from MLM_RoBERTa import MLM_RoBERTa
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class MLM_Roberta_POS_Tag(nn.Module):
    def __init__(self,model_path,train_set:str,test_set:str,num_pos_tag:int = 16):
        super(MLM_Roberta_POS_Tag,self).__init__()
        self.num_pos_tag = num_pos_tag
        self.roberta_model = MLM_RoBERTa(vocab_size=32000,ff_dim=768)
        self.roberta_model.load_state_dict(torch.load(model_path)) # we retrieve the weights and output sizes of pre_trained model 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classification_head = nn.Linear(32000, self.num_pos_tag)
        
        self.sp = spm.SentencePieceProcessor()
        model_path = 'm_model.model'
        self.sp.load(model_path)
        
        self.training_data = self.read_conll_file(file_path=train_set) # get a dictionnary containing the text and the POS tag for each word
        self.testing_data = self.read_conll_file(file_path=test_set) # get a dictionnary containing the text and the POS tag for each word

        self.optimizer = torch.optim.AdamW(self.roberta_model.parameters(), lr=1e-4)
    
    def forward(self,inputs): 
        input_ids = torch.as_tensor(self.sp.encode_as_ids(inputs)).to(self.device)
        # print(f'Input Shape: {input_ids.shape}')
        _, masked_logits, _ = self.roberta_model(input_ids,len(input_ids))
        pos_logits = self.classification_head(masked_logits)
        # print(f'Output Shape: {pos_logits.shape}')
        return pos_logits
    
    def train_pos_tag(self, loss_function, epochs=100):
        print(f'Starting fine-tuning training of MLM RoBERTa for POS-Tagging using {self.device}...')
        for epoch in range(epochs):
            total_loss = 0

            for data_point in self.training_data:
                inputs = data_point['text']
                # print(labels)

                labels_indices = data_point['labels']  # Use the precomputed labels_indices
                # Calculate the loss
                outputs = self(inputs) 
                
                # print(labels_tensor)
                loss = loss_function(outputs, labels_indices)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            average_loss = total_loss / len(self.training_data['text'])
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}")

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
