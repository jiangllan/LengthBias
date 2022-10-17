import os
import string

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch.nn import Softmax, Sigmoid

TASKS_AND_METRICS = {
    "qqp": ['acc', 'ba', 'tpr', 'tnr', 'pos_score', 'neg_score'],
    "twitter-url": ['F1', 'pre', 'recall'],
    "trecqa": ['mrr', "map"],
    "microblog": ['map', "p@30"]
}


def get_delta_lens(dataset, attr_list, mode):
    size = len(dataset[attr_list[0]])
    text_a, text_b = dataset[attr_list[0]], dataset[attr_list[1]]
    text_a_len = []
    text_b_len = []
    rel_delta_lens = []
    abs_delta_lens = []
    for i in range(size):
        a = text_a[i].translate(str.maketrans('', '', string.punctuation))
        b = text_b[i].translate(str.maketrans('', '', string.punctuation))
        len_a = len(a.split())
        len_b = len(b.split())
        abs_t = abs(len_a - len_b)
        rel_t = abs_t / min(len_a, len_b)
        rel_delta_lens.append(rel_t)
        abs_delta_lens.append(abs_t)
        text_a_len.append(len_a)
        text_b_len.append(len_b)
    if mode == "rel":
        return text_a_len, text_b_len, rel_delta_lens
    else:
        return text_a_len, text_b_len, abs_delta_lens


def partition_by_q(samples):
    all_q = np.array(list(set(samples['text_a'])))
    delta_lens_by_q = []
    for q in all_q:
        pos_docs = samples[(samples['text_a'] == q) & (samples['labels'] == 1)]
        neg_docs = samples[(samples['text_a'] == q) & (samples['labels'] == 0)]
        pos_avg_lens = pos_docs['delta_lens'].mean()
        neg_avg_lens = neg_docs['delta_lens'].mean()
        rel_len = abs(pos_avg_lens - neg_avg_lens) / min(pos_avg_lens, neg_avg_lens)
        delta_lens_by_q.append(rel_len)

    intervals = np.percentile(delta_lens_by_q, [0, 25, 50, 75, 100])
    intervals[-1] += 1
    partitions = []
    for i in range(intervals.size - 1):
        index = (delta_lens_by_q >= intervals[i]) & (delta_lens_by_q < intervals[i + 1])
        inr_q = all_q[index]
        inr_samples = samples[samples['text_a'].isin(inr_q)]
        partitions.append(inr_samples)

    return partitions


def sort_for_auc(preds, labels):
    Z = zip(preds, labels)
    Z = sorted(Z, reverse=True)
    preds_new, labels_new = zip(*Z)
    return preds_new, labels_new


def remove_outlier(data):
    percentile = np.percentile(data, [0, 25, 50, 75, 100])
    iqr = percentile[3] - percentile[1]
    up_limit = percentile[3] + iqr * 1.5
    down_limit = percentile[1] - iqr * 1.5
    return (data >= down_limit) & (data <= up_limit)


def avg_score(softmax_predictions, label_ids):
    positive_examples = softmax_predictions[np.array(label_ids) == 1]
    negative_examples = softmax_predictions[np.array(label_ids) == 0]
    pos_avg_scores = np.mean(positive_examples)
    neg_avg_scores = np.mean(negative_examples)
    return pos_avg_scores, neg_avg_scores


def split_sentence(sentence):
    sentence = sentence.replace('"', '').lower()
    word_list = sentence.split(' ')
    return word_list


def get_overlap(text_a, text_b):
    overlaps = []
    for i in range(len(text_b)):
        text_a_list = split_sentence(text_a[i])
        text_b_list = split_sentence(text_b[i])
        overlap = len(list(set(text_a_list) & set(text_b_list)))
        overlaps.append(overlap / min(len(text_a_list), len(text_b_list)))
    overlaps = np.array(overlaps)
    return overlaps


def MAP(samples, at=5):
    samples['rank'] = samples.groupby(by=['text_a'])['softmax_pred'].rank('max', ascending=False)
    rel_doc_nums = samples.groupby(by=['text_a'])['labels'].sum()
    rel_doc_order = np.concatenate(([np.arange(1, k + 1) for k in rel_doc_nums]))
    rel_doc_pos = np.sort(samples[samples['labels'] == 1]['rank'])
    tmp = rel_doc_order / rel_doc_pos
    AP_list = [np.mean(tmp[np.sum(rel_doc_nums[:i]): np.sum(rel_doc_nums[:i + 1])]) for i in range(rel_doc_nums.size)]
    mean_AP = np.mean(AP_list)
    return mean_AP


def MRR(samples):
    samples['rank'] = samples.groupby(by=['text_a'])['softmax_pred'].rank('max', ascending=False)
    rel_doc_nums = samples.groupby(by=['text_a'])['labels'].sum()
    rel_doc_pos = samples[samples['labels'] == 1]['rank']
    first_rel_pos = [np.min(rel_doc_pos[np.sum(rel_doc_nums[:i]): np.sum(rel_doc_nums[:i + 1])]) for i in
                     range(rel_doc_nums.size)]
    tmp = np.reciprocal(np.array(first_rel_pos))
    mrr = np.mean(tmp)
    return mrr


def metrics_wrt_delta_len(attr_list, test_data, preds, label_ids, file_dir, args):
    _, _, delta_lens = get_delta_lens(test_data, attr_list, "rel")
    dev_rel_intervals = np.loadtxt(os.path.join(file_dir, "dev.inr"))
    percentiles = np.concatenate(([-1], dev_rel_intervals[1:4], [1000]))

    pred_labels = np.argmax(preds, axis=1)
    # for auc
    stm = Softmax(dim=1)
    softmax_preds = stm(torch.from_numpy(preds))[:, 1].numpy()
    sgm = Sigmoid()
    delta_score = preds[:, 1] - preds[:, 0]
    sigmoid_preds = sgm(torch.from_numpy(delta_score)).numpy()
    df = pd.DataFrame({'text_a': test_data[attr_list[0]], 'text_b': test_data[attr_list[1]], 'delta_lens': delta_lens,
                       'pred_labels': pred_labels, 'softmax_pred': softmax_preds, #'sigmoid_pred': sigmoid_preds,
                       'labels': label_ids})

    # calculate in whole dataset and intervals
    result = {"all":{}}
    auc = metrics.roc_auc_score(label_ids, softmax_preds)
    if args.my_task.lower() == "twitter-url":
        macro_f1 = metrics.f1_score(label_ids, pred_labels, average="macro")
        micro_f1 = metrics.f1_score(label_ids, pred_labels, average="micro")
        result['all'].update({"macro-F1": round(macro_f1,4), "micro-F1": round(micro_f1,4)})
    if args.my_task.lower() in ["trecqa", "trec-2013"]:
        map, mrr = MAP(df), MRR(df)
        result['all'].update({"MAP": round(map,4), "MRR": round(mrr,4)})
    if args.my_task.lower() == "qqp":
        tn, fp, fn, tp = metrics.confusion_matrix(label_ids, pred_labels).ravel()
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        pos_rate = (tp + fp) / (tn + fn)
        # ppv = tp / (tp + fp)
        # npv = tn / (tn + fn)
        ba = (tpr + tnr) / 2
        acc = sum(pred_labels == label_ids) / len(test_data[attr_list[0]])
        pos_avg_confidence, neg_avg_confidence = avg_score(softmax_preds, label_ids)
        result['all'].update({"PosRatio": round(pos_rate, 4), "TPR": round(tpr, 4), "TNR": round(tnr, 4)})
        # result['all'].update({"PosRatio": round(pos_rate, 4), "TPR": round(tpr, 4), "TNR": round(tnr, 4),
        #                       "BA": round(ba, 4), "ACC": round(acc, 4), "pos_conf": round(pos_avg_confidence, 4),
        #                       "neg_conf": round(neg_avg_confidence, 4)})
    if args.my_task.lower() in ["trecqa", "trec-2013"]:
        partitions_by_q = partition_by_q(df)
        for i in range(len(partitions_by_q)):
            key = "INR-{}".format(str(i+1))
            samples = partitions_by_q[i]
            # print("case num: ", len(samples))
            map, mrr = MAP(samples), MRR(samples)
            result[key] = {"MAP": map, "MRR": mrr}
    else:
        for i in range(1, len(percentiles)):
            low, high = percentiles[i - 1], percentiles[i]
            samples = df[(df['delta_lens'] >= low) & (df['delta_lens'] < high)]
            print("case num: ", len(samples))
            try:
                assert (len(samples) > 0)
            except AssertionError:
                print(low, high)
            samples_labels = samples['labels'].to_numpy()
            pred_labels = samples['pred_labels'].to_numpy()
            # acc
            pred_correct = (pred_labels == samples_labels)
            # auc-method 1
            stm_preds = samples['softmax_pred'].to_numpy()
            # auc = metrics.roc_auc_score(samples_labels, stm_preds)
            # balanced acc
            key = str(round(low, 2)) + "～" + str(round(high, 2))
            result[key] = {}
            if args.my_task.lower() == "twitter-url":
                macro_f1 = metrics.f1_score(samples_labels, pred_labels, average="macro")
                micro_f1 = metrics.f1_score(samples_labels, pred_labels, average="micro")
                result[key].update({"macro-F1": round(macro_f1, 4), "micro-F1": round(micro_f1, 4)})
            if args.my_task.lower() in ["trecqa", "trec-2013"]:
                map, mrr = MAP(samples), MRR(samples)
                result[key].update({"MAP": round(map, 4), "MRR": round(mrr, 4)})
            if args.my_task.lower() == "qqp":
                acc = sum(pred_correct) / len(samples)
                tn, fp, fn, tp = metrics.confusion_matrix(samples_labels, pred_labels).ravel()
                tpr = tp / (tp + fn)
                tnr = tn / (tn + fp)
                pos_rate = (tp + fp) / (tn + fn)
                # ppv = tp / (tp + fp)
                # npv = tn / (tn + fn)
                ba = (tpr + tnr) / 2
                pos_avg_confidence, neg_avg_confidence = avg_score(stm_preds, samples_labels)
                result[key].update({"PosRatio": round(pos_rate, 4), "TPR": round(tpr, 4), "TNR": round(tnr, 4)})
                # result[key].update({"PosRatio": round(pos_rate, 4), "TPR": round(tpr, 4), "TNR": round(tnr, 4),
                #                     "BA": round(ba, 4), "ACC": round(acc, 4), "pos_conf": round(pos_avg_confidence, 4),
                #                     "neg_conf": round(neg_avg_confidence, 4)})
    return result
