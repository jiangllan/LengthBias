# On Length Divergence Bias in Textual Matching Models

This repo contains the code of our ACL 2022 paper, [On Length Divergence Bias in Textual Matching Models](https://aclanthology.org/2022.findings-acl.330.pdf).

## Reference

If you find our project useful in your research, please consider citing:

```
@inproceedings{jiang-etal-2022-length,
    title = "On Length Divergence Bias in Textual Matching Models",
    author = "Jiang, Lan and Lyu, Tianshu and Lin, Yankai and Chong, Meng and Lyu, Xiaoyong and Yin, Dawei",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    year = "2022",
}
```

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
4. [Contacts](#contacts)

## Introduction

In this work, we provide a new perspective to study this issue — via the length divergence bias. 
We find that:
1) length divergence heuristic widely exists in prevalent TM datasets, providing direct cues for prediction.
2) TM models have adopted such heuristic, and such bias can be attributed in part to extracting the text length information during training.

To alleviate the length divergence bias, we propose an adversarial training method.

## Usage

### Requirements

```bash
conda create -n length python=3.6
conda activate length

git clone https://github.com/jiangllan/LengthBias.git
cd LengthBias && pip install -r requirements.txt
```

### Datasets and Preparation

To obtain the balanced datasets, please run the `process.py` in `dataset`

```bash
cd dataset
python process.py
```
Finally, the `./dataset` foler will have teh following structure:

```angular2html
  '|-- dataset',
  '    |-- process.py',
  '    |-- Microblog',
  '    |   |-- dev.csv',
  '    |   |-- train.csv',
  '    |   |-- balanced',
  '    |       |-- dev.csv',
  '    |       |-- dev.inr',
  '    |       |-- train.csv',
  '    |       |-- train.inr',
  '    |-- QQP',
  '    |   |-- dev.csv',
  '    |   |-- train.csv',
  '    |   |-- balanced',
  '    |       |-- dev.csv',
  '    |       |-- dev.inr',
  '    |       |-- train.csv',
  '    |       |-- train.inr',
  '    |-- TrecQA',
  '    |   |-- dev.csv',
  '    |   |-- train.csv',
  '    |   |-- balanced',
  '    |       |-- dev.csv',
  '    |       |-- dev.inr',
  '    |       |-- train.csv',
  '    |       |-- train.inr',
  '    |-- Twitter-URL',
  '        |-- dev.csv',
  '        |-- train.csv',
  '        |-- balanced',
  '            |-- dev.csv',
  '            |-- dev.inr',
  '            |-- train.csv',
  '            |-- train.inr',
```

### Training and Evaluation

Coming soon...

## Contacts

jiangl20 at mails dot tsinghua dot edu dot cn