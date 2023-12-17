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


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, RobertaForTokenClassification, AdamW, RobertaConfig
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class PosTaggingDataset(Dataset):
    def __init__(self, data, tokenizer, label_mapping):
        self.data = data
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        labels = [self.label_mapping.get(label, self.label_mapping['<UNK>']) for label in self.data[idx]['labels']]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        # Pad labels to match the length of input_ids
        labels = labels + [self.label_mapping['<UNK>']] * (64 - len(labels))
        labels = torch.tensor(labels)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
class CollateFn:
    def __init__(self, tokenizer, label_mapping, max_length=64):
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        self.max_length = max_length

    def __call__(self, batch):
        input_ids_list = [item['input_ids'] for item in batch]
        texts = [self.tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in input_ids_list]

        # Find the actual maximum length in the batch
        max_actual_length = max(len(self.tokenizer.encode(text, add_special_tokens=True)) for text in texts)
        actual_max_length = min(self.max_length, max_actual_length)

        # Tokenize and truncate sequences
        inputs = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=actual_max_length,
            return_tensors='pt'
        )
        
        labels = torch.tensor([
            [self.label_mapping.get(label, self.label_mapping['<UNK>']) for label in item['labels'][:actual_max_length]] +
            [self.label_mapping['<UNK>']] * (actual_max_length - len(item['labels'][:actual_max_length]))
            for item in batch
        ])

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels
        }
    

class ConlluReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_file(self):
        data = []
        with open(self.file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            sentence = {'text': '', 'labels': []}
            for line in lines:
                if line.startswith('# text'):
                    sentence['text'] = line.split('=')[1].strip()
                elif line.startswith('#'):
                    # Ignore lines starting with '#', which are comments
                    continue
                elif line != '\n':
                    columns = line.split('\t')
                    if len(columns) >= 4:
                        label = columns[3]
                        sentence['labels'].append(label)
                    else:
                        print(f"Ignoring line: {line}")
                else:
                    data.append(sentence)
                    sentence = {'text': '', 'labels': []}
        return data