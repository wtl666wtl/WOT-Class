GPU=$1
dataset_name=$2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
model_name_or_path=bert-base-cased

# this is also defined in utils.py, make sure to change both when changing.
output_dir=../models/${model_name_or_path}

CUDA_VISIBLE_DEVICES=$GPU python train_text_classifier.py \
  --model_name_or_path ${model_name_or_path} \
  --task_name ${dataset_name} \
  --train_suffix "selected" \
  --test_suffix "test" \
  --output_dir ${output_dir} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 256 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --logging_steps 100000 \
  --save_steps -1 \
  --overwrite_output_dir \
  --overwrite_cache
