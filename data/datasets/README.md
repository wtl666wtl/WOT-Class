# Datasets

Please download our datasets from [Google Drive](https://drive.google.com/drive/folders/1YJti7o0sJtFfw8oiEr0BLr3xsBdao9xm?usp=drive_link) to this dir. There are 7 processed datasets as we introduced in our paper.



# How to add a new dataset
We split our train and test stage, so if you want to run our method on a new dataset, make sure there should be two folders `{new_dataset}_train` for train and `{new_dataset}_test` for test.

In `{new_dataset}_train` folder, you should provide

* `dataset.txt`: texts for training
* `labels.txt`: corresponding labels (for unlabeled text, the label should be n+1, where n is the maximum known label)
* `known_classes.txt`: known class names, should correspond to `labels.txt`
* `true_labels.txt`: gold labels, only for evaluation during training
* `entity2id.txt`, `sentences.json`: preprocessed files for the entity expansion, please refer to [CGExpan](https://github.com/yzhan238/CGExpan) or [HiExpan](https://github.com/mickeysjm/HiExpan) 

In `{new_dataset}_test` folder, you should provide

* `dataset.txt`: texts for testing
* `labels.txt`: gold labels, only for evaluation

