import numpy as np
from tqdm import tqdm
import json
import mmap
import copy
import os
import re
import spacy
import torch
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, f1_score
from collections import defaultdict as ddict
from scipy.optimize import linear_sum_assignment

DATA_FOLDER_PATH = os.path.join('..', 'data', 'datasets')
INTERMEDIATE_DATA_FOLDER_PATH = os.path.join('..', 'data', 'intermediate_data')
FINETUNE_MODEL_PATH = os.path.join('..', 'models')


def cosine_similarity_embeddings(emb_a, emb_b):
    return np.dot(emb_a, np.transpose(emb_b)) / np.outer(np.linalg.norm(emb_a, axis=1), np.linalg.norm(emb_b, axis=1))


def load_any_string(data_dir):
    with open(data_dir, mode='r', encoding='utf-8') as text_file:
        text = list(map(lambda x: x.strip(), text_file.readlines()))
    return text


def load_any_int(data_dir):
    with open(data_dir, mode='r', encoding='utf-8') as text_file:
        text = list(map(lambda x: int(x.strip()), text_file.readlines()))
    return text


def load_text(data_dir):
    with open(os.path.join(data_dir, 'dataset.txt'), mode='r', encoding='utf-8') as text_file:
        text = list(map(lambda x: x.strip(), text_file.readlines()))
    return text


def load_labels(data_dir):
    with open(os.path.join(data_dir, 'labels.txt'), mode='r', encoding='utf-8') as label_file:
        labels = list(map(lambda x: int(x.strip()), label_file.readlines()))
    return labels


def load_classnames(data_dir):
    with open(os.path.join(data_dir, 'classes.txt'), mode='r', encoding='utf-8') as classnames_file:
        class_names = "".join(classnames_file.readlines()).strip().split("\n")
    return class_names


def load_labels_num(data_dir):
    with open(os.path.join(data_dir, 'labels.txt'), mode='r', encoding='utf-8') as label_file:
        labels = list(map(lambda x: int(x.strip()), label_file.readlines()))
    return np.max(labels) + 1


def load_embedding(dataset_name):
    text = {}
    data_dir = os.path.join(DATA_FOLDER_PATH, dataset_name)
    with open(os.path.join(data_dir, '200d.txt'), mode='r', encoding='utf-8') as embedding_file:
        i = 0
        for x in embedding_file.readlines():
            i = i + 1
            if i == 1: continue
            p = x.strip().split(" ")
            t = []
            for j in range(200):
                t.append(float(p[j + 1]))
            text[p[0]] = t
    return text


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def load_sentences(data_dir):
    total_line = get_num_lines(os.path.join(data_dir, 'sentences.json'))
    lemma = []
    pos = []
    L = []
    P = []
    lastId = ""
    with open(os.path.join(data_dir, 'sentences.json'), 'r', encoding='utf-8') as sentences_file:
        for line in tqdm(sentences_file, total=total_line):
            obj = json.loads(line)
            if lastId != obj["articleId"]:
                if lastId != "":
                    lemma.append(L)
                    pos.append(P)
                    L = []
                    P = []
                lastId = obj["articleId"]
            L.extend(obj["lemma"])
            P.extend(obj["pos"])
        if lastId != "":
            lemma.append(L)
            pos.append(P)
    return lemma, pos


def text_statistics(text, name="default"):
    sz = len(text)

    tmp_text = [s.split(" ") for s in text]
    tmp_list = [len(doc) for doc in tmp_text]
    len_max = max(tmp_list)
    len_avg = np.average(tmp_list)
    len_std = np.std(tmp_list)

    print(f"\n### Dataset statistics for {name}: ###")
    print('# of documents is: {}'.format(sz))
    print('Document max length: {} (words)'.format(len_max))
    print('Document average length: {} (words)'.format(len_avg))
    print('Document length std: {} (words)'.format(len_std))
    print(f"#######################################")


def load_train_data(dataset_name):
    train_dataset_name = f"{dataset_name}_train"
    data_dir = os.path.join(DATA_FOLDER_PATH, train_dataset_name)
    class_names = load_any_string(os.path.join(data_dir, "known_classes.txt"))
    cleaned_text = load_any_string(os.path.join(data_dir, "dataset.txt"))
    label = load_any_int(os.path.join(data_dir, "labels.txt"))
    true_label = load_any_int(os.path.join(data_dir, "true_labels.txt"))
    lemma, pos = load_sentences(data_dir)

    result = {
        "class_names": class_names,
        "label": label,
        "cleaned_text": cleaned_text,
        "lemma": lemma,
        "pos": pos,
    }
    return result, true_label


def clean_str(string):
    string = clean_html(string)
    string = clean_email(string)
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def clean_email(string: str):
    return " ".join([s for s in string.split() if "@" not in s])


def clean_html(string: str):
    left_mark = '&lt;'
    right_mark = '&gt;'
    # for every line find matching left_mark and nearest right_mark
    while True:
        next_left_start = string.find(left_mark)
        if next_left_start == -1:
            break
        next_right_start = string.find(right_mark, next_left_start)
        if next_right_start == -1:
            print("Right mark without Left: " + string)
            break
        # print("Removing " + string[next_left_start: next_right_start + len(right_mark)])
        clean_html.clean_links.append(string[next_left_start: next_right_start + len(right_mark)])
        string = string[:next_left_start] + " " + string[next_right_start + len(right_mark):]
    return string


clean_html.clean_links = []

def tensor_to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


def cosine_similarity_embeddings_(emb_a, emb_b):
    return np.dot(emb_a, np.transpose(emb_b)) / np.outer(np.linalg.norm(emb_a), np.linalg.norm(emb_b))

def distance(emb_a, emb_b):
    dis = [[] for _ in range(len(emb_a))]
    for i in range(len(emb_a)):
        for j in range(len(emb_b)):
            dis[i].append(np.linalg.norm(emb_b[j]-emb_a[i], ord=2))
    return dis

def distance_(emb_a, emb_b):
    return np.linalg.norm(emb_b-emb_a, ord=2)

def evaluate_predictions(true_class, predicted_class, known_num=0, output_to_console=True):
    true_cls_num = np.max(true_class) + 1
    cls_num = np.max(predicted_class) + 1

    num = 0
    cnt = [0 for _ in range(cls_num)]
    for i in range(len(predicted_class)):
        if cnt[predicted_class[i]] == 0:
            cnt[predicted_class[i]] = 1
            num += 1
    print("number of classes in preds: " + str(num))

    if num == cls_num:
        confusion = confusion_matrix(true_class, predicted_class)
    else:
        max_len = max(true_cls_num, cls_num)
        confusion = np.zeros((max_len, max_len))
        for i in range(len(predicted_class)):
            confusion[true_class[i], predicted_class[i]] += 1

    print("-" * 80 + "Evaluating" + "-" * 80)
    print(confusion)

    #print(confusion)
    row_ind, col_ind = linear_sum_assignment(confusion[known_num:, known_num:].max() - confusion[known_num:, known_num:])
    c = min(cls_num, true_cls_num)
    true = copy.deepcopy(true_class)
    predicted = copy.deepcopy(predicted_class)
    if cls_num > true_cls_num:
        print(known_num, cls_num, true_cls_num)
        p = [0 for _ in range(cls_num)]
        a = np.argmax(confusion, axis=0)
        for i in range(known_num):
            p[i] = i
        for i in range(true_cls_num - known_num):
            p[known_num + col_ind[i]] = known_num + row_ind[i]
        for i in range(true_cls_num - known_num, min(cls_num - known_num, len(col_ind))):
            p[known_num + col_ind[i]] = a[known_num + col_ind[i]]
        for i in range(len(predicted_class)):
            predicted[i] = p[predicted[i]]
    elif cls_num < true_cls_num:
        p = [0 for _ in range(true_cls_num)]
        #a = np.argmax(confusion[known_num:, known_num:], axis=1)
        for i in range(known_num):
            p[i] = i
        for i in range(true_cls_num - known_num):
            p[known_num + col_ind[i]] = known_num + row_ind[i]
        #for i in range(cls_num - known_num, true_cls_num - known_num):
        #    p[known_num + row_ind[i]] = known_num + a[row_ind[i]]
        for i in range(len(predicted_class)):
            predicted[i] = p[predicted[i]]
    else:
        p = [0 for _ in range(cls_num)]
        for i in range(known_num):
            p[i] = i
        for i in range(true_cls_num - known_num):
            p[known_num + col_ind[i]] = known_num + row_ind[i]
        for i in range(len(predicted_class)):
            predicted[i] = p[predicted[i]]
    confusion_ = confusion_matrix(true, predicted)
    if output_to_console:
        print("-" * 80 + "Assignment" + "-" * 80)
        print(confusion_)

    mi = f1_score(true, predicted, average="micro")
    ma = f1_score(true, predicted, average="macro")
    s_mi = f1_score(true, predicted, labels=[_ for _ in range(known_num)], average="micro")
    s_ma = f1_score(true, predicted, labels=[_ for _ in range(known_num)], average="macro")
    un_mi = f1_score(true, predicted, labels=[_ for _ in range(known_num, c)], average="micro")
    un_ma = f1_score(true, predicted, labels=[_ for _ in range(known_num, c)], average="macro")
    if output_to_console:
        print("overall micro F1 score: " + str(mi))
        print("overall macro F1 score: " + str(ma))
        print("seen micro F1 score: " + str(s_mi))
        print("seen macro F1 score: " + str(s_ma))
        print("unseen micro F1 score: " + str(un_mi))
        print("unseen macro F1 score: " + str(un_ma))


def load_vocab(filename):
    eid2name = {}
    keywords = []
    name2eid = {}
    with open(filename, encoding='utf-8') as f:
        for line in f:
            temp = line.strip().split('\t')
            eid = int(temp[1])
            eid2name[eid] = temp[0]
            name2eid[temp[0]] = eid
            keywords.append(eid)
    eid2idx = {w:i for i, w in enumerate(keywords)}
    print(f'Vocabulary: {len(keywords)} keywords loaded')
    return eid2name, name2eid, keywords, eid2idx


def sentence_encode(tokens_id, model, layer):
    input_ids = torch.tensor([tokens_id], device=torch.device('cuda'))

    with torch.no_grad():
        hidden_states = model(input_ids)
    all_layer_outputs = hidden_states[2]

    layer_embedding = tensor_to_numpy(all_layer_outputs[layer].squeeze(0))[1: -1]
    return layer_embedding


def sentence_to_wordtoken_embeddings(layer_embeddings, tokenized_text, tokenized_to_id_indicies):
    word_embeddings = []
    for text, (chunk_index, start_index, end_index) in zip(tokenized_text, tokenized_to_id_indicies):
        word_embeddings.append(np.average(layer_embeddings[chunk_index][start_index: end_index], axis=0))
    assert len(word_embeddings) == len(tokenized_text)
    return np.array(word_embeddings)


def handle_sentence(model, layer, tokenized_text, tokenized_to_id_indicies, tokenids_chunks):
    layer_embeddings = [
        sentence_encode(tokenids_chunk, model, layer) for tokenids_chunk in tokenids_chunks
    ]
    word_embeddings = sentence_to_wordtoken_embeddings(layer_embeddings,
                                                       tokenized_text,
                                                       tokenized_to_id_indicies)
    return word_embeddings


def average_with_harmonic_series(representations):
    weights = [0.0] * len(representations)
    for i in range(len(representations)):
        weights[i] = 1. / (i + 1)
    return np.average(representations, weights=weights, axis=0)


def rank_by_significance(embeddings, class_embeddings):
    similarities = cosine_similarity_embeddings(embeddings, class_embeddings)
    significance_score = [np.max(softmax(similarity)) for similarity in similarities]
    significance_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(significance_score)))}
    return significance_ranking


def rank_by_relation(embeddings, class_embeddings):
    relation_score = cosine_similarity_embeddings(embeddings, [np.average(class_embeddings, axis=0)]).reshape((-1))
    relation_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(relation_score)))}
    return relation_ranking


def mul(l):
    m = 1
    for x in l:
        m *= x + 1
    return m


def weights_from_ranking(rankings):
    if len(rankings) == 0:
        assert False
    if type(rankings[0]) == type(0):
        rankings = [rankings]
    rankings_num = len(rankings)
    rankings_len = len(rankings[0])
    assert all(len(rankings[i]) == rankings_len for i in range(rankings_num))
    total_score = []
    for i in range(rankings_len):
        total_score.append(mul(ranking[i] for ranking in rankings))

    total_ranking = {i: r for r, i in enumerate(np.argsort(np.array(total_score)))}
    if rankings_num == 1:
        assert all(total_ranking[i] == rankings[0][i] for i in total_ranking.keys())
    weights = [0.0] * rankings_len
    for i in range(rankings_len):
        weights[i] = 1. / (total_ranking[i] + 1)
    return weights


def weight_sentence_with_attention(static_representations, contextualized_representations,
                                   class_representations, attention_mechanism):
    significance_ranking = rank_by_significance(contextualized_representations, class_representations)
    relation_ranking = rank_by_relation(contextualized_representations, class_representations)
    significance_ranking_static = rank_by_significance(static_representations, class_representations)
    relation_ranking_static = rank_by_relation(static_representations, class_representations)
    if attention_mechanism == "mixture":
        weights = weights_from_ranking((significance_ranking,
                                        relation_ranking,
                                        significance_ranking_static,
                                        relation_ranking_static))
    elif attention_mechanism == "significance_mixture":
        weights = weights_from_ranking((significance_ranking,
                                        significance_ranking_static))
    elif attention_mechanism == "significance":
        weights = weights_from_ranking(significance_ranking)
    elif attention_mechanism == "relation":
        weights = weights_from_ranking(relation_ranking)
    elif attention_mechanism == "significance_static":
        weights = weights_from_ranking(significance_ranking_static)
    elif attention_mechanism == "relation_static":
        weights = weights_from_ranking(relation_ranking_static)
    elif attention_mechanism == "none":
        weights = [1.0] * len(contextualized_representations)
    else:
        assert False
    return np.average(contextualized_representations, weights=weights, axis=0)
