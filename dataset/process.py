#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/18 21:26
# @Author  : Lan Jiang
# @File    : data_process.py

import os

import numpy as np
import pandas as pd

import pdb


def read_txt(file):
    with open(file, 'r') as f:
        data = f.readlines()
    data = [item.strip() for item in data]
    return data


def get_delta_lens(dataset):
    size = len(dataset)
    text_a, text_b = dataset['sentence1'].to_numpy(), dataset['sentence2'].to_numpy()
    text_a_len = []
    text_b_len = []
    rel_delta_lens = []
    for i in range(size):
        a = text_a[i]
        b = text_b[i]
        len_a = len(a.split())
        len_b = len(b.split())
        abs_t = abs(len_a - len_b)
        rel_t = abs_t / min(len_a, len_b)
        rel_delta_lens.append(rel_t)
        text_a_len.append(len_a)
        text_b_len.append(len_b)

    return rel_delta_lens


def split_index_list(delta_lens, intervals):
    delta_lens = np.array(delta_lens)
    size = len(intervals)
    result = [delta_lens < intervals[0]]
    for i in range(size - 1):
        result.append((delta_lens >= intervals[i]) & (delta_lens < intervals[i + 1]))
    result.append(delta_lens >= intervals[-1])
    return result


def rel_balanced_dataset(dataset, diff, intervals, std_ratio):
    total_cases = None
    indx_list = split_index_list(diff, intervals)
    for i, inds in enumerate(indx_list):
        print("\t[CAT %d]" % (i+1))
        cases = dataset[inds]
        pos_cases = cases[cases['label'] == 1]
        neg_cases = cases[cases['label'] == 0]
        print("\t**BEFORE** pos: {} neg: {} total: {} ratio: {:.2}".format(len(pos_cases), len(neg_cases), len(cases),
                                                                         len(pos_cases) / len(cases)))
        if len(pos_cases) / len(neg_cases) > std_ratio:
            std_pos_num = int(len(neg_cases) * std_ratio)
            reserved_pos_indx = np.random.choice([i for i in range(len(pos_cases))], std_pos_num, replace=False)
            pos_cases = pos_cases.iloc[reserved_pos_indx]
        elif len(pos_cases) / len(neg_cases) < std_ratio:
            std_neg_num = int(len(pos_cases) / std_ratio)
            reserved_neg_indx = np.random.choice([i for i in range(len(neg_cases))], std_neg_num, replace=False)
            neg_cases = neg_cases.iloc[reserved_neg_indx]
        cases = np.append(pos_cases, neg_cases, axis=0)
        if total_cases is None:
            total_cases = cases
        else:
            total_cases = np.append(total_cases, cases, axis=0)
        print("\t**AFTER** pos: {} neg: {} total: {} ratio: {:.2}".format(len(pos_cases), len(neg_cases), len(cases),
                                                                          len(pos_cases) / len(cases)))
    return total_cases


def write_data(cases, adv_dir):
    data = pd.DataFrame({'label': cases[:, 0],
                         'uid': cases[:, 1],
                         'sentence1': cases[:, 2],
                         'sentence2': cases[:, 3]
                         })
    data = data.sample(frac=1)
    data.to_csv(os.path.join(adv_dir, "%s.csv" % split), index=False)


if __name__ == "__main__":
    data_dir = "./"
    for task in ["QQP", "Twitter-url", "TrecQA", "Microblog"]:
        print("\n"+"="*12, "Processing Task %s" % task, "="*12)
        adv_dir = os.path.join(data_dir, task, "adversarial")
        if not os.path.exists(adv_dir):
            os.mkdir(adv_dir)
        for split in ["train", "dev"]:
            print("[Split %s]" % split)
            file = os.path.join(data_dir, task, "%s.csv" % split)
            dataset = pd.read_csv(file)
            rel_lens = get_delta_lens(dataset)
            intervals = np.percentile(rel_lens, [0, 25, 50, 75, 100])
            np.savetxt(os.path.join(adv_dir, "%s.inr" % split), intervals, delimiter=",")
            label_ratio = len(dataset[dataset['label'] == 1]) / len(dataset[dataset['label'] == 0])
            rel_dataset = rel_balanced_dataset(dataset, rel_lens, intervals[1:4], label_ratio)
            write_data(rel_dataset, adv_dir)
        print("Processing %s data over." % task)
