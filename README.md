# WOT-Class: Weakly Supervised Open-world Text Classification

This repo contains the data and the code for the [paper](https://arxiv.org/pdf/2305.12401.pdf).

## Requirements
The python version we used is 3.7, also, you need to install torch. The other requirements are listed in requirements.txt.

## Data & Methods
The datasets can be accessed at `data/`.

The methods are implemented in `src/`.

To run our code on a dataset (e.g., AGNews), please first follow the guide of [CGExpan](https://github.com/yzhan238/CGExpan) or [HiExpan](https://github.com/mickeysjm/HiExpan) to process the raw data and get the `entity2id.txt` and `sentences.json` files under the training dataset file.

Then, you can train and test our method by calling `run.sh`.

```
cd src
sh run.sh {dataset_name} {gpu_id} {known_number_of_classes}
```

You can also modify `run.sh` to control the hyperparameters we support.
The performances on the datasets, and their behaviors when using different hyperparameters, can be found in our paper.

## Citation
If you find this repo useful, please cite our paper:
```
@article{Wang2023WOTClassWS,
  title={WOT-Class: Weakly Supervised Open-world Text Classification},
  author={Tianle Wang and Zihan Wang and Weitang Liu and Jingbo Shang},
  journal={ArXiv},
  year={2023},
  volume={abs/2305.12401}
}
```