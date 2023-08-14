set -e

dataset=$1
gpu=$2
known_class=$3

CUDA_VISIBLE_DEVICES=${gpu} python Embedding_processor.py --dataset ${dataset}
CUDA_VISIBLE_DEVICES=${gpu} python main.py --dataset_name ${dataset}
sh run_train_text_classifier.sh ${gpu} ${dataset}
python evaluate.py --dataset ${dataset} --stage final