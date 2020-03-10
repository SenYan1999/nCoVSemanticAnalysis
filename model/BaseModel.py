import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BaseBertModel(nn.Module):
    def __init__(self, bert_type, d_bert, num_class):
        super(BaseBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_type)
        self.classifier = nn.Linear(d_bert, num_class)

    def forward(self, x):
        x_mask = (x != 1).int()
        x_output = self.bert(x, attention_mask=x_mask)[1]
        x_output = self.classifier(x_output)

        return torch.log_softmax(x_output, dim=-1)