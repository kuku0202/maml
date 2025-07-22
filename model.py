import torch
import torch.nn as nn
from transformers import BertModel, AutoModel

class ProteinTransformer(nn.Module):
    """
    A protein transformer model that builds on top of a pretrained language model.
    The model extracts the [CLS] token output and passes it through an MLP head to predict a property.
    """
    def __init__(self, num_outputs=1, dropout=0.1, hidden_size=256, model_name='prot_bert'):
        super(ProteinTransformer, self).__init__()
        self.model_name = model_name
        
        if model_name == 'bert-base-uncased':
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        elif model_name == 'prot_bert':
            self.bert = BertModel.from_pretrained('Rostlab/prot_bert')
        elif model_name == 'esm2':
            self.bert = AutoModel.from_pretrained('facebook/esm2_t6_8M_UR50D')
        else:
            print(f"Unknown model name {model_name}")

        self.predictor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_outputs)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        prediction = self.predictor(cls_output)
        return prediction