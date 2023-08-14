import numpy as np
import pandas as pd
import torch, gc
from torch import nn
from torch import optim
import transformers as tfs
import math
import os

from Bert_classification_model import Bert_classification_model
from torch.utils.data import Dataset

batch_size = 64
model_class, tokenizer_class, pretrained_weights = (tfs.BertModel, tfs.BertTokenizer, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
sos_id, eos_id = tokenizer.encode("", add_special_tokens=True)


class TextDataset(Dataset):
    def __init__(self, labeled, text, label=None):
        self.text = text
        self.labeled = labeled
        self.label = label
        self.batch_tokenized = tokenizer.batch_encode_plus(text, add_special_tokens=True,
                                                           max_length=64, truncation=True, pad_to_max_length=True)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = (torch.tensor(self.batch_tokenized["input_ids"][item]),
                torch.tensor(self.batch_tokenized["attention_mask"][item]))
        if self.label:
            return text, self.label[item]
        else:
            return text, []


class Rough_classifier(object):

    def __init__(self, inputs, targets, dataset, k, epochs=5, lr=5e-6):
        self.train_inputs = inputs
        self.train_targets = targets
        self.dataset = dataset

        self.batch_size = batch_size
        self.batch_count = len(self.train_inputs) // self.batch_size
        self.batch_count += (int)(len(self.train_inputs) % self.batch_size > 0)
        labeled_set = TextDataset(labeled=True, text=inputs, label=targets)
        if k > 0:
            self.loader = torch.utils.data.DataLoader(labeled_set, batch_size=batch_size, shuffle=True,
                                                  num_workers=4)
            self.rep_loader = torch.utils.data.DataLoader(labeled_set, batch_size=batch_size, shuffle=False,
                                                      num_workers=4)
        self.epochs = epochs
        self.lr = lr
        self.k = k  # class num
        self.device = torch.device('cuda')
        self.bert_classifier_model = Bert_classification_model(self.k).to(self.device)
        self.optimizer = optim.Adam(self.bert_classifier_model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def get_model(self):
        return self.bert_classifier_model

    def set_model(self, model):
        self.bert_classifier_model = model

    def train(self):
        self.bert_classifier_model.train()
        gc.collect()
        torch.cuda.empty_cache()
        for epoch in range(self.epochs):
            print_avg_loss = 0
            for batch_idx, ((x, mx), label) in enumerate(self.loader):
                x = x.to(self.device)
                mx = mx.to(self.device)
                label = label.to(self.device)
                self.optimizer.zero_grad()
                outputs, _, _ = self.bert_classifier_model(x, mx)
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()
                print_avg_loss += loss.item()
            print("Epoch: %d, Loss: %.4f" % ((epoch + 1), print_avg_loss / self.batch_count))

    def get_represent(self):
        self.bert_classifier_model.eval()
        rep = []
        with torch.no_grad():
            for batch_idx, ((x, mx), _) in enumerate(self.rep_loader):
                x = x.to(self.device)
                mx = mx.to(self.device)
                _, r, _ = self.bert_classifier_model(x, mx)
                for j in range(r.size(0)):
                    rep.append(r[j].detach().cpu().numpy().reshape(-1).tolist())
        return rep

    def get_text_represent(self, text):
        self.bert_classifier_model.eval()
        text_set = TextDataset(labeled=False, text=text)
        text_loader = torch.utils.data.DataLoader(text_set, batch_size=batch_size, shuffle=False,
                                                  num_workers=4)
        rep = []
        with torch.no_grad():
            for batch_idx, ((x, mx), _) in enumerate(text_loader):
                x = x.to(self.device)
                mx = mx.to(self.device)
                _, r, _ = self.bert_classifier_model(x, mx)
                for j in range(r.size(0)):
                    rep.append(r[j].detach().cpu().numpy().reshape(-1).tolist())
        return rep
