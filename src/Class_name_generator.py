import numpy as np
import os
import tqdm
from collections import defaultdict as ddict
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from Expan import *
from utils import *

stops = set(stopwords.words('english'))

def judge(s):
    return s == "NN" or s == "NNS" or s == "NNP" or s == "NNPS"

class Class_name_generator(object):

    def __init__(self, N, dataset_name, dataset, HOT_WORDS_NUM=50):
        self.HOT_WORDS_NUM = HOT_WORDS_NUM
        self.N = N
        self.candidates = ddict(int)
        self.vocab = {}
        self.dataset_name = f"{dataset_name}_train"
        self.lemma = dataset["lemma"]
        self.pos = dataset["pos"]
        self.dataset = dataset
        self.word_num = [ddict(int) for _ in range(self.N)]
        self.word_appear = [ddict(int) for _ in range(self.N)]
        for id in range(self.N):
            for i in range(len(self.lemma[id])):
                w = self.lemma[id][i].lower()
                if len(w) < 2 or (not w.isalpha()) or w in stops or (not judge(self.pos[id][i])): continue
                self.vocab[w] = 1
        for id in range(self.N):
            flag = ddict(int)
            for i in range(len(self.lemma[id])):
                w = self.lemma[id][i].lower()
                if w not in self.vocab: continue
                if judge(self.pos[id][i]):
                    self.word_num[id][w] += 1
                    flag[w] = 1
                self.word_num[id][w] += 1
            for w in flag:
                self.word_appear[id][w] += 1

    def generate_class_names(self, expand_num=200, load_from_file=False):
        query_set = []
        res = []
        self.expan = Expan(self.dataset_name, torch.device("cuda"))
        for i in range(len(self.dataset["class_names"])):
            if self.dataset["class_names"][i] in self.expan.name2eid:
                query_set.append(self.expan.name2eid[self.dataset["class_names"][i]])
        if not load_from_file:
            expanded = self.expan.expand(query_set, expand_num)
            filename = 'expand.json'
            with open(filename, 'w') as f:
                json.dump(expanded, f)
        else:
            filename = 'expand.json'
            print("Loading...")
            with open(filename) as f:
                expanded = json.load(f)
            print("OK!")
        for eid in expanded:
            res.append(self.expan.eid2name[eid].lower())
        return res

    def get_hot_words(self, cls_num):
        tf = [ddict(int) for _ in range(cls_num)]
        in_time = [ddict(int) for _ in range(cls_num)]
        total_time = ddict(int)
        num = [0 for _ in range(cls_num)]
        for id in range(self.N):
            cls = self.dataset["cluster"][id]
            if cls == -1:
                for w in self.word_num[id]:
                    total_time[w] += 1
                continue
            num[cls] += 1
            for w in self.word_appear[id]:
                total_time[w] += 1
                in_time[cls][w] += 1

        for id in range(self.N):
            cls = self.dataset["cluster"][id]
            if cls == -1: continue
            for w in self.word_num[id]:
                if w in in_time[cls]:
                    tf[cls][w] += self.word_num[id][w]

        hws = []
        for cls in range(cls_num):
            aux = []
            for w in tf[cls]:
                val = (in_time[cls][w] / num[cls]) * np.tanh(tf[cls][w] / num[cls]) * np.log(self.N / total_time[w])
                aux.append((val, w))
            aux.sort(reverse=True)
            hot_words = [w for key, w in aux[0:self.HOT_WORDS_NUM]]
            hws.append(hot_words)
        return hws

    def get_hot_words_for_known(self, cls_num):
        tf = [ddict(int) for _ in range(cls_num)]
        in_time = [ddict(int) for _ in range(cls_num)]
        total_time = ddict(int)
        num = [0 for _ in range(cls_num)]
        for id in range(self.N):
            cls = self.dataset["label"][id]
            if cls == cls_num:
                for w in self.word_num[id]:
                    total_time[w] += 1
                continue
            num[cls] += 1
            for w in self.word_appear[id]:
                total_time[w] += 1
                in_time[cls][w] += 1

        for id in range(self.N):
            cls = self.dataset["label"][id]
            if cls == cls_num: continue
            for w in self.word_num[id]:
                if w in in_time[cls]:
                    tf[cls][w] += self.word_num[id][w]

        hws = []
        for cls in range(cls_num):
            aux = []
            for w in tf[cls]:
                val = (in_time[cls][w] / num[cls]) * np.tanh(tf[cls][w] / num[cls]) * np.log(self.N / total_time[w])
                aux.append((val, w))
            aux.sort(reverse=True)
            hot_words = [w for key, w in aux[0:self.HOT_WORDS_NUM]]
            hws.append(hot_words)
        return hws
