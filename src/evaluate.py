import argparse
import json
import os
import pickle

import numpy as np

from utils import *


def evaluate(dataset, stage, known_num, suffix=None):
    train_dataset_name = f"{dataset}_train"
    test_dataset_name = f"{dataset}_test"
    data_dir = os.path.join(DATA_FOLDER_PATH, train_dataset_name)
    gold_labels = load_any_int(os.path.join(data_dir, "true_labels.txt"))
    if stage == "Rep":
        with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, dataset, f"document_repr.pk"), "rb") as f:
            dictionary = pickle.load(f)
            document_representations = dictionary["document_representations"]
            class_representations = dictionary["class_representations"]
            repr_prediction = np.argmax(cosine_similarity_embeddings(document_representations, class_representations),
                                        axis=1)
            evaluate_predictions(gold_labels, repr_prediction, known_num=known_num)
    elif stage == "Align":
        with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, dataset, f"data.pk"), "rb") as f:
            dictionary = pickle.load(f)
            documents_to_class = dictionary["documents_to_class"]
            evaluate_predictions(gold_labels, documents_to_class, known_num=known_num)
    else:
        data_dir = os.path.join(DATA_FOLDER_PATH, test_dataset_name)
        gold_labels = load_any_int(os.path.join(data_dir, "labels.txt"))
        with open(os.path.join(FINETUNE_MODEL_PATH, "bert-base-cased", "eval_labels.json"), "r") as f:
            pred_labels = json.load(f)
            evaluate_predictions(gold_labels, pred_labels, known_num=known_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--stage", type=str)
    args = parser.parse_args()
    print(vars(args))

    know_class_names = load_any_string(os.path.join(DATA_FOLDER_PATH, f"{args.dataset}_train/known_classes.txt"))
    know_num = len(know_class_names)

    evaluate(args.dataset, args.stage, know_num)
