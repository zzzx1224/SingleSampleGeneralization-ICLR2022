# Learning to Generalize across Domains on Single Test Samples
Code for paper "Learning to Generalize across Domains on Single Test Samples" published at ICLR 2022.

### Prerequisites
 - Python 3.7.9
 - Pytorch 1.7.1
 - GPU: a single NVIDIA Tesla V100

### Components
 - `../kfold224/`: Directory of images
 - `../files/`: Directory of the train/validation/test split txt files
 - `pacs_main.py`: script to run classification experiments on PACS
 - `pacs_model.py`: the model used in `pacs_main.py`
 - `pacs_dataset.py`: script to load data from PACS for the experiments
 - `pacs_rtdataset.py`: script to load source data from PACS to generate the classifier
 - `pacs_test.py`: script to evaluate the trained model
 - `./logs/`: folder to store the trained model
 - `augs.py`: data augmentation functions for the experiments 


## Setup
The code is for the PACS dataset. Download the datasets from the following link, extract the compressed file, and place the images in `../kfold224/` directory and the train/validation/test split txt files in the `../files/` directory.

[[Google Drive](https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk)] provided by Li et al. in [[Deeper, Broader and Artier Domain Generalization](https://arxiv.org/pdf/1710.03077v1.pdf)]

## Training
For training the model run the following:
```
python pacs_main.py --test_domain cartoon --log_dir MODEL_NAME
```
Change the `cartoon` after `--test_domain` to `art_painting`, `photo` or `sketch` to change the target domain.
Use `--net res18/res50` to choose ResNet-18 or ResNet-50 as the backbone. The default value is `--net res18`.
Use `--lr` to set the learning rate of the classifier generation module. The default value is `--lr 0.0001`.
Use `--reslr` to set the learning rate of the backbone network. The default value is `--reslr 0.5`. The learning rate of the backbone equal to the multiplication of the value of `--lr` and the value of `--reslr`
Use `--ctx_num` to choose the number of samples per category per domain to generate the adapted classifier. The default value is `--ctx_num 10`.
The trained model and logs will be stored in `./logs/MODEL_NAME/`

## Evaluation
For evaluation of the trained model run the following:
```
python pacs_test.py --test_domain cartoon --log_dir res18_cartoon
```
Change the target domain as the training phase.
Save the trained model for cartoon domain in `./logs/res18_cartoon/`. Change the `res18_cartoon` to other names to evaluate other trained models
