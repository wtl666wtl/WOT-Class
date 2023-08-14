import argparse
import pickle as pk
from collections import Counter

import string
import numpy as np
import torch
import copy
import json
from utils import *
from Rough_classifier import *
from K_clustering import *
from Class_name_generator import *
from Fine_tuning_classifier import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    print(args)
    set_seed(args.random_state)

    dataset, true_label = load_train_data(args.dataset_name) # true_label only for eval
    class_names = copy.deepcopy(dataset["class_names"])

    know_num = len(dataset["class_names"])
    guess_num = args.initial_estimation

    print(f"initial estimation of classes: {guess_num}")
    print("Finish reading data!")

    # find potential class names (from CGExpan)
    CNG = Class_name_generator(len(dataset["cleaned_text"]), args.dataset_name, dataset, args.W)
    initial_candidate_class_names = CNG.generate_class_names(expand_num=args.initial_estimation)

    # get initial document representations from x-class
    dataset["represent"] = []
    RC_text = []
    RC_label = []
    for i in range(len(dataset["label"])):
        if dataset["label"][i] < know_num:
            RC_text.append(dataset["cleaned_text"][i])
            RC_label.append(dataset["label"][i])
    RC = Rough_classifier(RC_text, RC_label, dataset, know_num)
    RC.train()
    model = RC.get_model()
    print("Finish training!")

    initial_candidate_class_names.extend(dataset["class_names"])
    dataset["class_names"] = copy.deepcopy(initial_candidate_class_names)
    FTC = Fine_tuning_classifier(dataset, model, initial_candidate_class_names, args.dataset_name,
                                 know_num, args.alpha, args.beta, args.W_statistic, random_state=args.random_state)
    FTC.get_static_representation()
    FTC.get_class_oriented_document_representation_initial()

    # get initial clusters through initial document representation
    KC = K_clustering(dataset, know_num, guess_num, pca=512, random_state=args.random_state)
    dataset["cluster"] = KC.clustering()

    # merge classes (only if labeled texts from the same class are over half of two clusters)
    # that'ok to delete this part... seems not useful in few-shot setting
    cls_num = [0 for _ in range(guess_num)]
    for i in range(len(dataset["cleaned_text"])):
        cls_num[dataset["cluster"][i]] += 1
    for cls in range(know_num):
        center = [0 for _ in range(guess_num)]
        for i in range(len(dataset["cleaned_text"])):
            if dataset["label"][i] == cls:
                center[dataset["cluster"][i]] += 1
        c = center.index(max(center))
        max_ratio = 0
        if center[c] < 0.5 * cls_num[c]:
            for i in range(guess_num):
                if center[i] / cls_num[i] > max_ratio:
                    max_ratio = center[i] / cls_num[i]
                    c = i
        cnt = 0
        while cnt < guess_num:
            if center[cnt] >= 0.5 * cls_num[cnt] and cnt != c:
                guess_num -= 1
                for j in range(len(dataset["cluster"])):
                    if dataset["cluster"][j] == cnt:
                        dataset["cluster"][j] = c
                    if dataset["cluster"][j] == guess_num:
                        dataset["cluster"][j] = cnt

                cls_num = [0 for _ in range(guess_num)]
                center = [0 for x in range(guess_num)]
                for i in range(len(dataset["cleaned_text"])):
                    cls_num[dataset["cluster"][i]] += 1
                    if dataset["label"][i] == cls:
                        center[dataset["cluster"][i]] += 1
                cnt = -1
                c = center.index(max(center))
            cnt += 1
        for i in range(len(dataset["cleaned_text"])):
            if dataset["cluster"][i] == c:
                dataset["cluster"][i] = cls
            elif dataset["cluster"][i] == cls:
                dataset["cluster"][i] = c

    # eval after the initial process
    unlabel = []
    for i in range(len(dataset["label"])):
        if dataset["label"][i] == know_num:
            unlabel.append(i)
    ul_classes = [dataset["cluster"][i] for i in unlabel]
    ul_gold_classes = [true_label[i] for i in unlabel]
    evaluate_predictions(ul_gold_classes, ul_classes, know_num)

    print("Start iterative learning!")
    iter = 0
    while True:
        iter += 1
        print("Iter: " + str(iter))
        guess_num = 1 + np.max(dataset["cluster"])

        # iteratively remove redundant clusters
        removal_flag = False
        iter_cnt = iter
        while True:
            #  find statistically representative words
            dataset["class_hot_words"] = CNG.get_hot_words_for_known(know_num)
            dataset["class_hot_words"].extend(CNG.get_hot_words(guess_num))
            print(dataset["class_hot_words"])
            word_num = ddict(int)
            for i in range(guess_num + know_num):
                for w in dataset["class_hot_words"][i]:
                    word_num[w] += 1
            # alleviate noise
            for i in range(guess_num + know_num):
                for j in range(min(args.W, len(dataset["class_hot_words"][i])) - 1, -1, -1):
                    if word_num[dataset["class_hot_words"][i][j]] >= (guess_num + know_num) // 2:
                        dataset["class_hot_words"][i].pop(j)

            check_num = 0
            pseudo_cluster = [0 for _ in range(guess_num)]
            for i in range(know_num, know_num + guess_num):
                if len(dataset["class_hot_words"][i]) == 0:
                    pseudo_cluster[i-know_num] = -1
                else:
                    pseudo_cluster[i-know_num] = check_num
                    check_num += 1
            if check_num != guess_num:
                for i in range(len(dataset["cluster"])):
                    dataset["cluster"][i] = pseudo_cluster[dataset["cluster"][i]]
                guess_num = check_num
                continue

            # find potential class names (from statistically representative words)
            candidate_class_names = []
            for p in range(args.W_statistic):
                for i in range(know_num, know_num + guess_num):
                    if len(dataset["class_hot_words"][i]) == 0:
                        continue
                    flag = False
                    for w in dataset["class_hot_words"][i][:]:
                        if w in candidate_class_names:
                            flag = True
                            break
                    if flag: continue
                    candidate_class_names.append(dataset["class_hot_words"][i][p])
            candidate_class_names.extend(initial_candidate_class_names)

            dataset["class_names"] = copy.deepcopy(class_names)
            FTC = Fine_tuning_classifier(dataset, model, candidate_class_names, args.dataset_name, guess_num + know_num,
                                         args.alpha, args.beta, args.W_statistic, random_state=args.random_state)

            selected_names_set = FTC.score_potential_words(iter_cnt)
            print(f"class-words set: {selected_names_set}")

            # remove impure clusters by \eta in paper
            purity = []
            cls_document_representations = [[] for _ in range(guess_num)]
            for i in range(len(dataset["cluster"])):
                cls_document_representations[dataset["cluster"][i]].append(dataset["represent"][i])
            for cls in range(guess_num):
                ave = np.average(cls_document_representations[cls], axis=0)
                sim = 0.0
                for i in range(len(cls_document_representations[cls])):
                    sim += cosine_similarity_embeddings_(ave, cls_document_representations[cls][i])
                purity.append(sim / len(cls_document_representations[cls]))

            def purify_(purity, selected_names_set):
                num = know_num
                cls_num = [0 for _ in range(guess_num)]
                pseudo_cluster = [-1 for _ in range(guess_num)]
                names_set = [[] for _ in range(know_num)]
                for i in range(len(dataset["cleaned_text"])):
                    if dataset["cluster"][i] == -1: continue
                    cls_num[dataset["cluster"][i]] += 1
                for cls in range(know_num):
                    pseudo_cluster[cls] = cls
                    names_set[cls].append(class_names[cls])

                distance = []
                for cls in range(guess_num):
                    if pseudo_cluster[cls] > -1:
                        continue
                    distance.append((-purity[cls], cls))
                distance = sorted(distance)
                left_num = len(distance)

                def judge(num, cls):
                    last = -1
                    for i in range(num):
                        cnt = 0
                        for j in range(len(selected_names_set[cls])):
                            if selected_names_set[cls][j] in names_set[i]:
                                cnt += 1
                        if cnt > 0:
                            if last > -1:
                                return -2
                            else:
                                last = i
                    return last

                for i in range(left_num):
                    _, cls = distance[i]
                    if len(selected_names_set[cls]) == 0: continue
                    tmp = judge(num, cls)
                    if tmp > -1:
                        pseudo_cluster[cls] = -1
                    elif tmp == -1:
                        names_set.append(selected_names_set[cls][:])
                        pseudo_cluster[cls] = num
                        num += 1
                print(names_set)
                return num, pseudo_cluster, names_set

            guess_total_num, pseudo_cluster, potential_names = purify_(purity, selected_names_set)

            for i in range(len(dataset["cluster"])):
                if dataset["cluster"][i] == -1: continue
                dataset["cluster"][i] = pseudo_cluster[dataset["cluster"][i]]

            print("continue to remove clusters...")

            if guess_num == guess_total_num:
                break
            else:
                iter_cnt += 1
                removal_flag = True
                guess_num = guess_total_num

        if removal_flag == False and iter > 1:
            break

        dataset["class_names"] = copy.deepcopy(class_names)

        # update by x-class
        FTC_ = Fine_tuning_classifier(dataset, model, candidate_class_names, args.dataset_name, guess_total_num,
                                      args.alpha, args.beta, args.W_statistic,
                                      random_state=args.random_state, potential_names=potential_names)
        RC_text, RC_label = FTC_.work()

        # fientune with pseudo data
        RC_ = Rough_classifier(RC_text, RC_label, dataset, guess_total_num)
        RC_.train()
        model = RC_.get_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--initial_estimation", type=int, default=100, help="initial over-estimation of classes")
    parser.add_argument("--W_statistic", type=int, default=3, help="# of statistically representative words for potential class-words")
    parser.add_argument("--W", type=int, default=50, help="# of statistically representative words for MLP")
    parser.add_argument("--alpha", type=float, default=0.9, help="cutoff threshold of indicativeness score")
    parser.add_argument("--beta", type=float, default=0.7, help="cutoff threshold of indicativeness score")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(args)