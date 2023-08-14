import torch
from torch import nn
from torch import optim
import transformers as tfs
import math
import numpy as np


class Bert_classification_model(nn.Module):
    def __init__(self, k):
        super(Bert_classification_model, self).__init__()
        model_class, tokenizer_class, pretrained_weights = (tfs.BertModel, tfs.BertTokenizer, 'bert-base-uncased')
        self.bert = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dense = nn.Linear(768, k)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):
        _, pooled, hidden = self.bert(input_ids, attention_mask=attention_mask)
        linear_output = self.dense(self.dropout(pooled))
        return linear_output, pooled, hidden
