# modified from XClass: https://github.com/ZihanWangKi/XClass
import random

import numpy as np
import os
import tqdm
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import transformers as tfs
import math
import copy
import pickle as pk
from tqdm import tqdm
from shutil import copyfile
from scipy.special import softmax
from collections import Counter
from collections import defaultdict as ddict
import string
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from transformers import BertTokenizerFast, BertTokenizer, BertModel, BertForMaskedLM
from utils import *
from Rough_classifier import *
import faiss

Vocab_Min_Occurrence = 5
Top_K = 1
Layer = 12
Iteration = 100
Train_Iter = 100
Pca = 64
Confidence_threshold = 0.5
SELECTED_NAMES = 6
CARED_NAMES = 1

class toy_classifier(nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(toy_classifier, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, 1)

    def forward(self, x):
        hidden = F.relu(self.hidden(x))
        out = torch.sigmoid(self.out(hidden))
        return out


def prepare_sentence(tokenizer, text):
    # setting for BERT
    model_max_tokens = 512
    has_sos_eos = True
    ######################
    max_tokens = model_max_tokens
    if has_sos_eos:
        max_tokens -= 2
    sliding_window_size = max_tokens // 2

    tokenized_text = tokenizer.basic_tokenizer.tokenize(text, never_split=tokenizer.all_special_tokens)
    tokenized_to_id_indicies = []

    tokenids_chunks = []
    tokenids_chunk = []

    for index, token in enumerate(tokenized_text + [None]):
        if token is not None:
            tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        if token is None or len(tokenids_chunk) + len(tokens) > max_tokens:
            tokenids_chunks.append([sos_id] + tokenids_chunk + [eos_id])
            if sliding_window_size > 0:
                tokenids_chunk = tokenids_chunk[-sliding_window_size:]
            else:
                tokenids_chunk = []
        if token is not None:
            tokenized_to_id_indicies.append((len(tokenids_chunks),
                                             len(tokenids_chunk),
                                             len(tokenids_chunk) + len(tokens)))
            tokenids_chunk.extend(tokenizer.convert_tokens_to_ids(tokens))

    return tokenized_text, tokenized_to_id_indicies, tokenids_chunks


def find_words(class_words_representations, finished, static_word_representations, K, class_words, vocab_words,
               masked_words):
    finished_class = copy.deepcopy(finished)
    old_words = copy.deepcopy(masked_words)
    cls_repr = [None for _ in range(K)]
    for t in range(1, Iteration):
        class_representations = [average_with_harmonic_series(class_words_representation)
                                 for class_words_representation in class_words_representations]
        cosine_similarities = cosine_similarity_embeddings(static_word_representations,
                                                           class_representations)
        nearest_class = cosine_similarities.argmax(axis=1)
        similarities = cosine_similarities.max(axis=1)
        for cls in range(K):
            if cls in finished_class:
                continue
            highest_similarity = -1.0
            highest_similarity_word_index = -1
            lowest_masked_words_similarity = 1.0
            existing_class_words = set(class_words[cls])
            stop_criterion = False
            for i, word in enumerate(vocab_words):
                if nearest_class[i] == cls:
                    if word not in masked_words:
                        if similarities[i] > highest_similarity:
                            highest_similarity = similarities[i]
                            highest_similarity_word_index = i
                    else:
                        if word not in existing_class_words and word not in old_words:
                            # print(1, word, cls)
                            stop_criterion = True
                            break
                        if word in existing_class_words:
                            lowest_masked_words_similarity = min(lowest_masked_words_similarity, similarities[i])
                else:
                    if word in existing_class_words:
                        stop_criterion = True
                        break
            # the topmost t words are no longer the t words in class_words
            if lowest_masked_words_similarity < highest_similarity:
                stop_criterion = True

            if stop_criterion:
                finished_class.add(cls)
                cls_repr[cls] = average_with_harmonic_series(class_words_representations[cls])
                break
            class_words[cls].append(vocab_words[highest_similarity_word_index])
            class_words_representations[cls].append(static_word_representations[highest_similarity_word_index])
            masked_words.add(vocab_words[highest_similarity_word_index])
            cls_repr[cls] = average_with_harmonic_series(class_words_representations[cls])
        if len(finished_class) == K:
            break
    return masked_words, cls_repr


class Fine_tuning_classifier(object):

    def __init__(self, dataset, model, candidate_class_names, dataset_name, K,
                 alpha=0.9, beta=0.6, W_sta=3, know_num=-1 ,random_state=42, potential_names=None):
        self.dataset = dataset
        self.candidate_class_names = candidate_class_names
        self.origin_name = dataset_name
        self.dataset_name = f"{dataset_name}_train"
        self.data_folder = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, dataset_name)
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.W_sta = W_sta
        self.potential_names = potential_names
        if know_num == -1:
            self.know_num = len(self.dataset["class_names"])
        else:
            self.know_num = know_num
        self.K = K
        self.random_state = random_state

    def weight_sentence(self, tokenization_info, attention="mixture"):
        tokenized_text, tokenized_to_id_indicies, tokenids_chunks, contextualized_word_representations, \
        static_representations, contextualized_representations = tokenization_info
        if len(contextualized_representations) == 0:
            print("Empty Sentence (or sentence with no words that have enough frequency)")
            return np.average(contextualized_word_representations, axis=0)
        else:
            return weight_sentence_with_attention(static_representations, contextualized_representations,
                                                  self.class_representations, attention)

    def get_static_representation(self):
        model = self.model
        model.eval()
        model.cuda()

        counts = Counter()
        for text in tqdm(self.dataset["cleaned_text"]):
            tokenized_text, tokenized_to_id_indicies, tokenids_chunks = prepare_sentence(tokenizer, text)
            counts.update(word.translate(str.maketrans('', '', string.punctuation)) for word in tokenized_text)

        del counts['']
        updated_counts = {k: c for k, c in counts.items() if c >= Vocab_Min_Occurrence}
        word_rep = {}
        word_count = {}
        tokenization_info = []
        for text in tqdm(self.dataset["cleaned_text"]):
            tokenized_text, tokenized_to_id_indicies, tokenids_chunks = prepare_sentence(tokenizer, text)
            contextualized_word_representations = handle_sentence(model, Layer, tokenized_text,
                                                                  tokenized_to_id_indicies, tokenids_chunks)

            tokenization_info.append((tokenized_text, tokenized_to_id_indicies,
                                      tokenids_chunks, contextualized_word_representations))
            for i in range(len(tokenized_text)):
                word = tokenized_text[i]
                if word in updated_counts.keys():
                    if word not in word_rep:
                        word_rep[word] = 0
                        word_count[word] = 0
                    word_rep[word] += contextualized_word_representations[i]
                    word_count[word] += 1

        word_avg = {}
        for k, v in word_rep.items():
            word_avg[k] = word_rep[k] / word_count[k]

        vocab_words = list(word_avg.keys())
        static_word_representations = list(word_avg.values())
        vocab_occurrence = list(word_count.values())

        word_to_index = {v: k for k, v in enumerate(vocab_words)}

        new_info = []
        sentence_representations = []
        for i, _tokenization_info in tqdm(enumerate(tokenization_info), total=len(tokenization_info)):
            static_representations = []
            contextualized_representations = []
            tokenized_text, tokenized_to_id_indicies, tokenids_chunks, \
            contextualized_word_representations = _tokenization_info
            for i, token in enumerate(tokenized_text):
                if token in word_to_index:
                    static_representations.append(static_word_representations[word_to_index[token]])
                    contextualized_representations.append(contextualized_word_representations[i])
            new_info.append((tokenized_text, tokenized_to_id_indicies, tokenids_chunks,
                             contextualized_word_representations, static_representations, contextualized_representations))
            sentence_representations.append(contextualized_representations)

        os.makedirs(self.data_folder, exist_ok=True)
        with open(os.path.join(self.data_folder, f"tokenization.pk"), "wb") as f:
            pk.dump({
                "tokenization_info": new_info,
            }, f, protocol=4)

        with open(os.path.join(self.data_folder, f"static_repr.pk"), "wb") as f:
            pk.dump({
                "static_word_representations": static_word_representations,
                "vocab_words": vocab_words,
                "word_to_index": word_to_index,
                "vocab_occurrence": vocab_occurrence,
                "sentence_representations": sentence_representations
            }, f, protocol=4)
        print("Finish get_static_representation()!")

    def get_class_oriented_document_representation_initial(self):
        static_repr_path = os.path.join(self.data_folder, f"static_repr.pk")
        with open(static_repr_path, "rb") as f:
            vocab = pk.load(f)
            static_word_representations = vocab["static_word_representations"]
            word_to_index = vocab["word_to_index"]
            vocab_words = vocab["vocab_words"]
        with open(os.path.join(self.data_folder, f"tokenization.pk"), "rb") as f:
            tokenization_info = pk.load(f)["tokenization_info"]

        class_names_ = self.dataset["class_names"]
        class_names = []
        for name in class_names_:
            if name in word_to_index:
                class_names.append(name)

        class_words_representations = [static_word_representations[word_to_index[class_names[cls]]]
                                       for cls in range(len(class_names))]

        self.class_representations = np.array(class_words_representations)
        self.model.eval()
        self.model.cuda()
        self.vocab = vocab

        document_representations = []
        for i, _tokenization_info in tqdm(enumerate(tokenization_info), total=len(tokenization_info)):
            document_representations.append(self.weight_sentence(_tokenization_info, attention="significance_mixture"))

        document_representations = np.array(document_representations)
        self.dataset["represent"] = document_representations.tolist()
        print("Finish get_class_oriented_document_representation_initial()!")

    def get_class_oriented_document_representation_new(self):
        static_repr_path = os.path.join(self.data_folder, f"static_repr.pk")
        with open(static_repr_path, "rb") as f:
            vocab = pk.load(f)
            static_word_representations = vocab["static_word_representations"]
            word_to_index = vocab["word_to_index"]
            vocab_words = vocab["vocab_words"]
        with open(os.path.join(self.data_folder, f"tokenization.pk"), "rb") as f:
            tokenization_info = pk.load(f)["tokenization_info"]

        masked_words = set()
        class_words = []
        class_words_representations = []
        for i in range(len(self.potential_names)):
            class_words.append(self.potential_names[i])
            sum = 0
            for j in range(len(self.potential_names[i])):
                sum += static_word_representations[word_to_index[self.potential_names[i][j]]]
                masked_words.add(self.potential_names[i][j])
            class_words_representations.append([sum / len(self.potential_names[i])])

        finished_class = set()
        masked_words, cls_repr = find_words(class_words_representations, finished_class,
                                            static_word_representations, self.K, class_words,
                                            vocab_words, masked_words)

        self.class_representations = np.array(cls_repr)
        self.model.eval()
        self.model.cuda()
        self.vocab = vocab

        document_representations = []
        for i, _tokenization_info in tqdm(enumerate(tokenization_info), total=len(tokenization_info)):
            document_representations.append(self.weight_sentence(_tokenization_info))

        document_representations = np.array(document_representations)
        self.dataset["represent"] = document_representations.tolist()

        print("Finish get_class_oriented_document_representation()!")

    def document_class_alignment(self):
        print("Start document_class_alignment()!")
        document_representations = np.array(self.dataset["represent"]).astype('float32')
        class_representations = np.array(self.class_representations).astype('float32')
        if Pca:
            mat = faiss.PCAMatrix(768, Pca)
            mat.train(document_representations)
            assert mat.is_trained
            document_representations = mat.apply_py(document_representations)
            class_representations = mat.apply_py(class_representations)
            print("Finish PCA!")

        cosine_similarities = cosine_similarity_embeddings(document_representations, class_representations)
        document_class_assignment = np.argmax(cosine_similarities, axis=1)
        document_class_assignment_matrix = np.zeros((document_representations.shape[0], self.K))
        for i in range(document_representations.shape[0]):
                document_class_assignment_matrix[i][document_class_assignment[i]] = 1.0

        gmm = GaussianMixture(n_components=self.K, covariance_type='tied',
                              random_state=self.random_state,
                              n_init=999, warm_start=True)
        gmm.converged_ = "HACK"

        gmm._initialize(document_representations, document_class_assignment_matrix)
        gmm.lower_bound_ = -np.infty
        gmm.fit(document_representations)

        documents_to_class = gmm.predict(document_representations)
        centers = gmm.means_
        distance = -gmm.predict_proba(document_representations) + 1
        self.dataset["cluster"] = documents_to_class.tolist()

        with open(os.path.join(self.data_folder, f"data.pk"), "wb") as f:
            pk.dump({
                "documents_to_class": documents_to_class,
                "distance": distance,
            }, f, protocol=4)
        print("Finish document_class_alignment()!")

    def prepare(self):
        with open(os.path.join(self.data_folder, f"data.pk"), "rb") as f:
            save_data = pk.load(f)
            documents_to_class = save_data["documents_to_class"]
            distance = save_data["distance"]
            num_classes = distance.shape[1]
        pseudo_document_class_with_confidence = [[] for _ in range(num_classes)]
        for i in range(documents_to_class.shape[0]):
            pseudo_document_class_with_confidence[documents_to_class[i]].append((distance[i][documents_to_class[i]], i))

        selected = []
        for i in range(num_classes):
            pseudo_document_class_with_confidence[i] = sorted(pseudo_document_class_with_confidence[i])
            num_docs_to_take = int(len(pseudo_document_class_with_confidence[i]) * Confidence_threshold)
            confident_documents = pseudo_document_class_with_confidence[i][:num_docs_to_take]
            confident_documents = [x[1] for x in confident_documents]
            selected.extend(confident_documents)

        selected = sorted(selected)
        text = [self.dataset["cleaned_text"][i] for i in selected]
        classes = [documents_to_class[i] for i in selected]

        ### eval
        gold_labels = list(
            map(int, open(os.path.join(DATA_FOLDER_PATH, self.dataset_name, "true_labels.txt")).readlines()))
        gold_classes = [gold_labels[i] for i in selected]
        evaluate_predictions(gold_classes, classes, self.know_num)
        ###

        write_to_dir(text, classes, self.origin_name)
        print("Finish prepare()!")
        return text, classes

    def score_potential_words(self, iter):
        static_repr_path = os.path.join(self.data_folder, f"static_repr.pk")
        with open(static_repr_path, "rb") as f:
            vocab = pk.load(f)
            static_word_representations = vocab["static_word_representations"]
            word_to_index = vocab["word_to_index"]
            vocab_words = vocab["vocab_words"]

        can = []
        for c in self.candidate_class_names:
            if c in vocab_words and c not in can:
                can.append(c)

        KEYWORDS_NUM = 20
        SAMPLE_LIMIT = len(can) // 2
        SAMPLE_TIME = 5
        model = toy_classifier(4, 32)
        model.to("cuda")
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss = torch.nn.BCELoss()

        def build(x, y, z): # build features: distance & cos similarity
            sim = cosine_similarity_embeddings(y, [x])
            avg_1 = 0
            min_1 = 1
            max_1 = 0
            var_1 = 0
            tot = 0
            for i in range(len(y)):
                avg_1 += sim[i][0] / z[i]
                max_1 = max(max_1, sim[i][0])
                min_1 = min(min_1, sim[i][0])
                tot += 1.0 / z[i]
            avg_1 /= tot
            for i in range(len(y)):
                var_1 += (sim[i][0] - avg_1) * (sim[i][0] - avg_1) / z[i]
            var_1 /= tot

            avg_2 = 0
            min_2 = 1
            max_2 = 0
            var_2 = 0
            for i in range(len(y)):
                dis = np.linalg.norm(y[i] - x)
                avg_2 += dis / z[i]
                max_2 = max(max_2, dis)
                min_2 = min(min_2, dis)
            avg_2 /= tot
            for i in range(len(y)):
                var_2 += (sim[i][0] - avg_1) * (sim[i][0] - avg_1) / z[i]
            var_2 /= tot

            res = [avg_1, var_1, avg_2, var_2]
            return torch.unsqueeze(torch.Tensor(res), 0).to("cuda")

        keywords = [[] for _ in range(self.K)]
        num = ddict(int)
        for cls in range(self.K):
            for i in range(len(self.dataset["class_hot_words"][cls])):
                if self.dataset["class_hot_words"][cls][i] in vocab_words:
                    num[self.dataset["class_hot_words"][cls][i]] += 1
        for cls in range(self.K):
            for i in range(len(self.dataset["class_hot_words"][cls])):
                if self.dataset["class_hot_words"][cls][i] in vocab_words:
                    keywords[cls].append((static_word_representations[
                                              word_to_index[self.dataset["class_hot_words"][cls][i]]],
                                          num[self.dataset["class_hot_words"][cls][i]]))
        for it in range(Train_Iter):
            for cls in range(len(self.dataset["class_names"])):
                q = []
                for c in can:
                    dis = np.linalg.norm(static_word_representations[word_to_index[c]] - static_word_representations[
                        word_to_index[self.dataset["class_names"][cls]]])
                    q.append((-dis, c))
                q = sorted(q)
                for keywords_num in range(len(keywords[cls]), len(keywords[cls]) + 1):
                    cls_name = static_word_representations[word_to_index[self.dataset["class_names"][cls]]]
                    sample_keywords = []
                    sample_num = []
                    for i in range(keywords_num):
                        sample_keywords.append(keywords[cls][i][0])
                        sample_num.append(keywords[cls][i][1])
                    optim.zero_grad()
                    pre = model(build(cls_name, sample_keywords, sample_num))
                    lo = loss(pre, torch.ones(1, 1).to("cuda"))
                    lo.backward()
                    optim.step()
                    for t in range(SAMPLE_TIME):
                        optim.zero_grad()
                        t = random.randint(0, SAMPLE_LIMIT)
                        pre = model(build(static_word_representations[word_to_index[q[t][1]]], sample_keywords, sample_num))
                        lo = loss(pre, torch.zeros(1, 1).to("cuda"))
                        lo.backward()
                        optim.step()

        for cls in range(len(self.dataset["class_names"])):
            if self.dataset["class_names"][cls] not in can:
                can.append(self.dataset["class_names"][cls])
        model.eval()
        unknown_class_num = self.K - len(self.dataset["class_names"])
        sim_names = [[] for _ in range(unknown_class_num)]
        for c in can:
            key = []
            sum = 0
            with torch.no_grad():
                for cls in range(len(self.dataset["class_names"]), self.K):
                    sum_v = 0
                    for keywords_num in range(len(keywords[cls]), len(keywords[cls]) + 1):
                        sample_keywords = []
                        sample_num = []
                        for i in range(keywords_num):
                            sample_keywords.append(keywords[cls][i][0])
                            sample_num.append(keywords[cls][i][1])
                        sum_v += model(build(static_word_representations[word_to_index[c]], sample_keywords, sample_num)).item()
                    v = sum_v
                    key.append(v)
                    sum += v
            for cls in range(unknown_class_num):
                val = -key[cls]
                sim_names[cls].append((val, val / (sum - key[cls]), c))
        selected_names_set = [[] for _ in range(unknown_class_num)]
        max_val = []
        for cls in range(unknown_class_num):
            sim_names[cls] = sorted(sim_names[cls])
            max_val.append(sim_names[cls][0][0])
        for w in can:
            position = []
            sum_p = 0
            for cls in range(unknown_class_num):
                for j in range(len(sim_names[cls])):
                    if sim_names[cls][j][2] == w:
                        position.append(j)
                        sum_p += 1.0 - j / len(sim_names[cls])
                        break
            for cls in range(unknown_class_num):
                for j in range(len(sim_names[cls])):
                    if sim_names[cls][j][2] == w:
                        sim_names[cls][j] = (sim_names[cls][j][0] * np.log((np.median(position) + 1) / (j + 2)),
                                             sim_names[cls][j][0], w)
                        break
        for cls in range(unknown_class_num):
            sim_names[cls] = sorted(sim_names[cls])
            cared_names_iter = iter
            for j in range(cared_names_iter):
                if sim_names[cls][j][1] <= self.alpha * sim_names[cls][0][1] and sim_names[cls][j][0] \
                        <= self.beta * sim_names[cls][0][0] and len(selected_names_set[cls]) < cared_names_iter:
                    selected_names_set[cls].append(sim_names[cls][j][2])
        for i in range(self.K - unknown_class_num):
            self.dataset["class_hot_words"].pop(0)
        return selected_names_set

    def work(self):
        self.get_class_oriented_document_representation_new()
        self.document_class_alignment()
        text, classes = self.prepare()
        print("Finish work()!")
        return text, classes # pseudo labels


def write_to_dir(text, labels, dataset_name):
    assert len(text) == len(labels)
    train_dataset_name = f"{dataset_name}_train"
    new_dataset_name = f"{dataset_name}_selected"
    # removes all potentially cached files in it
    # if os.path.isdir(os.path.join(DATA_FOLDER_PATH, new_dataset_name)):
    # assert False, f"{os.path.join(DATA_FOLDER_PATH, new_dataset_name)} exists."
    os.makedirs(os.path.join(DATA_FOLDER_PATH, new_dataset_name), exist_ok=True)
    with open(os.path.join(DATA_FOLDER_PATH, new_dataset_name, "dataset.txt"), "w") as f:
        for i, line in enumerate(text):
            f.write(line)
            f.write("\n")

    with open(os.path.join(DATA_FOLDER_PATH, new_dataset_name, "labels.txt"), "w") as f:
        for i, line in enumerate(labels):
            f.write(str(line))
            f.write("\n")
    copyfile(os.path.join(DATA_FOLDER_PATH, train_dataset_name, "known_classes.txt"),
             os.path.join(DATA_FOLDER_PATH, new_dataset_name, "classes.txt"))
