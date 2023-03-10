# AIdea-2022-fall
AIdea 農地作物現況調查影像辨識競賽-秋季賽：AI作物影像判釋
## Requirement
+ OS: ubuntu 22.04
+ Nvidia GPU with CUDA version 11.7
+ Anaconda
## Environment Installation
Create an new conda virtual environment
```
conda create -n convnext python=3.8 -y
conda activate AIdea
```
Install required packages:
```
pip install torch torchvision torchaudio
pip install timm albumentations yacs wandb xgboost
```
Download code:
```
git clone https://github.com/jason122490/AIdea-2022-fall
```
## Data Preparation
Train data structure:
```
├── dataset/train/
│   ├── class1/
│   |   ├── img001.jpg
│   |   ├── img002.jpg
│   |   └── ....
│   ├── class2/
│   |   ├── img001.jpg
│   |   ├── img002.jpg
│   |   └── ....
│   └── ....
└──
```
Public test data structure:
```
├── dataset/public_test/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ....
└──
```
Private test data structure:
```
├── dataset/private_test/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ....
└──
```
+ We provide image resizing tool for image resize to speed up the training speed in `/tool`
## Train and Infer step:
### 1. Training
Start training by running:
```
model/ConvNeXt_large.ipynb
model/ConvNeXt_base.ipynb
model/EfficientNetV2_large.ipynb
model/Swin_transformerV2_base.ipynb
```
model weight will be save in this structure:
```
├── model/model_name/
│   ├── checkpoint/
│   |   ├── ckpt001.pt
│   |   ├── ckpt002.pt
│   |   └── ....
│   ├── best_acc/
│   |   ├── acc_90.pt
│   |   ├── acc_92.pt
│   |   └── ....
│   └── 
└──
```
NOTICE: if you only want to train, but not output embedding data, just run jupyter notebook until Start Training section.
### 2. Pseudo label
Generate pseudo label by running:
```
ConvNeXt_large_get_pseudo_label.ipynb
```
pseudo label will be save in this structure:
```
├── dataset/pseudo_label/
│   ├── class1/
│   |   ├── img001.jpg
│   |   ├── img002.jpg
│   |   └── ....
│   ├── class2/
│   |   ├── img001.jpg
│   |   ├── img002.jpg
│   |   └── ....
│   └── ....
└──
```
### 3. Training with pseudo label
Start training with pseudo label and output embedding data for xgboost by running:
```
model/ConvNeXt_large.ipynb
model/ConvNeXt_base.ipynb
model/EfficientNetV2_large.ipynb
model/Swin_transformerV2_base.ipynb
```
model weight will be save in this structure:
```
├── model/model_name/
│   ├── checkpoint/
│   |   ├── ckpt001.pt
│   |   ├── ckpt002.pt
│   |   └── ....
│   ├── best_acc/
│   |   ├── acc_90.pt
│   |   ├── acc_92.pt
│   |   └── ....
│   └── 
└──
```
embedding data(image features) for xgboost will be save in this structure:
```
├── model/
│   ├── train_emb_model_name
│   └── test_emb_model_name
└──
```
NOTICE: need to change config in file to pseudo train
### 4. Ensemble and XGBoost
Start to ensemble four model and train xgboost to produce the final result by running:
```
model/Ensemble_XGBoost.ipynb
```
private and public's prediction will be save as:
```
model/test.csv
```
