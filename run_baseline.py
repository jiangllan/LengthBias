#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/12 8:03 下午
# @Author  : Lan Jiang
# @File    : run_baseline.py

import argparse
import logging

import numpy as np
import os
import torch
from progressbar import *
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.BiMPM import BiMPM
from models.ESIM import ESIM
from models.MatchPyramid import MatchPyramid
from utils.analysis_utils import metrics_wrt_delta_len
from utils.data_utils import BaselineDataset, characterize, positional_encoding

logger = logging.getLogger()


def prepare_inputs(batch, args):
    s1, s2 = torch.stack(batch[0], dim=1), torch.stack(batch[1], dim=1)
    # limit the lengths of input sentences up to max_sent_len
    if len(s1[0]) > args.max_len:
        s1 = s1[:, :args.max_len]
    if len(s2[0]) > args.max_len:
        s2 = s2[:, :args.max_len]
    kwargs = {'text_a': s1, 'text_b': s2, 'pos_ids': batch[2],
              "text_a_len": batch[3], "text_b_len": batch[4]}
    if args.model_name.lower() == "bimpm" and args.use_char_emb:
        char_p = Variable(torch.LongTensor(characterize(s1, args.characterized_words)))
        char_h = Variable(torch.LongTensor(characterize(s2, args.characterized_words)))
        kwargs['char_a'] = char_p
        kwargs['char_b'] = char_h
    label = torch.LongTensor([int(item) for item in batch[-1]]).to(args.device)
    for k, v in kwargs.items():
        kwargs[k] = v.to(args.device)

    return kwargs, label


def train(epoch, model, train_iter, eval_iter, optimizer, criterion, best_eval_acc, args):
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)
    model.train()
    loss = 0.0

    for i, batch in enumerate(tqdm(train_iter, desc="Train")):
        # prepare input for model
        inputs, label = prepare_inputs(batch, args)
        optimizer.zero_grad()
        preds = model(**inputs)
        batch_loss = criterion(preds, label)
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.detach().cpu()
        if (len(train_iter) * epoch + i + 1) % args.logging_step == 0:
            # do eval
            eval_result, preds = evaluate(model, eval_iter, criterion, args)
            print("***** Eval results {} *****".format(args.my_task))
            for key, value in eval_result.items():
                print("  {} = {}".format(key, value))
            # save best model
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            # if best_eval_loss[0] is None or eval_result['loss'] < best_eval_loss[0]:
            if best_eval_acc[0] is None or eval_result['acc'] > best_eval_acc[0]:
                torch.save(model, os.path.join(args.save_dir, "{}.pt".format(args.model_name)))
                # best_eval_loss[0] = eval_result['loss']
                best_eval_acc[0] = eval_result['acc']
                print("save best model to: ", os.path.join(args.save_dir, "{}.pt".format(args.model_name)))
            model.train()

    train_loss = loss / len(train_iter)
    return train_loss


def evaluate(model, eval_iter, criterion, args):
    model.eval()
    correct_num = 0
    loss = []
    preds = None
    for i, batch in enumerate(tqdm(eval_iter, desc="Eval")):
        inputs, label = prepare_inputs(batch, args)
        batch_preds = model(**inputs)
        batch_loss = criterion(batch_preds, label)
        batch_correct_num = (torch.argmax(batch_preds, dim=1) == label).sum().detach().cpu()
        correct_num += batch_correct_num
        loss.append(batch_loss.detach().cpu())
        if preds is None:
            preds = batch_preds.detach().cpu()
        else:
            preds = torch.cat([preds, batch_preds.detach().cpu()], dim=0)

    dev_acc = float(correct_num) / args.test_size
    dev_loss = np.mean(loss)
    eval_result = {"acc": dev_acc, "loss": dev_loss}
    return eval_result, preds


def for_visual(model, eval_iter, criterion, args):
    model.eval()
    correct_num = 0
    loss = []
    preds = None
    eb_a = None
    eb_b = None
    fh = None
    for i, batch in enumerate(tqdm(eval_iter, desc="Eval")):
        inputs, label = prepare_inputs(batch, args)
        batch_preds, batch_eb_a, batch_eb_b, batch_fh = model(**inputs)
        batch_loss = criterion(batch_preds, label)
        batch_correct_num = (torch.argmax(batch_preds, dim=1) == label).sum().detach().cpu()
        correct_num += batch_correct_num
        loss.append(batch_loss.detach().cpu())

        preds = batch_preds.detach().cpu() if preds is None else torch.cat([preds, batch_preds.detach().cpu()], dim=0)
        eb_a = batch_eb_a.detach().cpu() if eb_a is None else torch.cat([eb_a, batch_eb_a.detach().cpu()], dim=0)
        eb_b = batch_eb_b.detach().cpu() if eb_b is None else torch.cat([eb_b, batch_eb_b.detach().cpu()], dim=0)
        fh = batch_fh.detach().cpu() if fh is None else torch.cat([fh, batch_fh.detach().cpu()], dim=0)

    return preds, eb_a, eb_b, fh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="BiMPM",
                        help='baseline model, includes: ESIM, BiMPM, DIIN.')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='where to load pre-trained model file.')
    parser.add_argument('--data_dir', default="../Dataset/glue/", type=str,
                        help='where to load data files.')
    parser.add_argument('--embedding_dir', default="../Dataset/", type=str,
                        help='where to load embedding files.')
    parser.add_argument('--save_dir', default="../tmp/baseline", type=str,
                        help='where to load data files.')
    parser.add_argument('--analysis_mode', default="abs", type=str,
                        help="Used in evaluation, options: [rel, abs]")
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    parser.add_argument('--logging_step', default=1500, type=int)
    parser.add_argument('--my_task', default='MRPC', help='available: MRPC or Quora')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epoch_num', default=10, type=int)
    parser.add_argument('--word_vocab_size', default=50000, type=int)
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--linear_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--max_len', default=50, type=int,
                        help='max length of input sentences model can accept, if -1, it accepts any length')
    parser.add_argument('--num_perspective', default=20, type=int)
    parser.add_argument('--use_pos_emb', default=False, action='store_true')
    parser.add_argument('--balanced_mode', default="", type=str,
                        help="options: [ratio_balanced, abs_balanced]")
    parser.add_argument('--delta_lens_mode', default="", type=str,
                        help="options: [rel_delta_lens, abs_delta_lens]")
    parser.add_argument('--skip_header', default=False, action='store_true')
    parser.add_argument('--do_train', default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')
    parser.add_argument('--word_dim', default=300, type=int)

    args = parser.parse_args()

    # set seed for reproducibility
    torch.manual_seed(0)  # set for cpu
    torch.cuda.manual_seed_all(0)  # set for all gpu

    # set device
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.device_ids)
    args.device = torch.device(args.device_id)
    args.save_dir = os.path.join(args.save_dir, args.my_task, args.delta_lens_mode, args.balanced_mode,
                                 args.model_name.lower(), "use_pos_emb" if args.use_pos_emb else "")

    # load data
    train_dataset = BaselineDataset(args, split="train")
    train_iter = DataLoader(train_dataset, batch_size=args.train_batch_size)
    if args.do_eval:
        eval_dataset = BaselineDataset(args, split="dev")
        eval_iter = DataLoader(eval_dataset, batch_size=args.eval_batch_size)
        args.test_size = len(eval_dataset)
    args.characterized_words = train_dataset.characterized_words
    setattr(args, 'word_vocab_size', train_dataset.max_words)
    setattr(args, 'class_size', len(train_dataset.label_list))
    setattr(args, 'max_word_len', train_dataset.max_word_len)
    if args.use_char_emb:
        setattr(args, 'char_vocab_size', len(train_dataset.characterized_words))
    print("load data over.")

    # load model
    model = None
    weights_matrix = torch.from_numpy(train_dataset.weights_matrix)
    pos_weights_matrix = None
    if args.use_pos_emb:
        d_model = args.word_dim if args.model_name.lower() == "matchpyramid" else args.hidden_size
        pos_weights_matrix = torch.from_numpy(positional_encoding(args.max_len, d_model)[0])
    if args.model_name.lower() == "esim":
        model = ESIM(args, weights_matrix, pos_weights_matrix)
    elif args.model_name.lower() == "matchpyramid":
        model = MatchPyramid(args, weights_matrix, pos_weights_matrix)
    elif args.model_name.lower() == "bimpm":
        model = BiMPM(args, weights_matrix, pos_weights_matrix)
    model.to(args.device)
    print("load {} model over.".format(args.model_name))
    print("#Parameters: {} #Trainable Parameters: {}".format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # do train
    criterion = nn.CrossEntropyLoss()
    if args.do_train:
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(parameters, lr=args.learning_rate)
        best_eval_acc = [None]
        # best_eval_loss = [None]
        print("Start training...")
        for epoch in range(args.epoch_num):
            train_loss = train(epoch, model, train_iter, eval_iter, optimizer, criterion, best_eval_acc, args)

    # training finished
    if args.do_eval:
        pre_trained_model = args.model_dir if args.model_dir is not None else args.save_dir
        print("load model from: ", pre_trained_model)
        model = torch.load(os.path.join(pre_trained_model, "{}.pt".format(args.model_name)))
        eval_result, preds = evaluate(model, eval_iter, criterion, args)
        label_ids = [int(item) for item in eval_dataset.dataset['label']]
        file_dir = os.path.join(args.data_dir, args.my_task)
        analysis = metrics_wrt_delta_len(['text_a', 'text_b'], eval_dataset.dataset, preds.numpy(), label_ids, file_dir,
                                         args)
        print("***** Eval results {} *****".format(args.my_task))
        for key, value in eval_result.items():
            print("  {} = {}".format(key, value))
        print("***** Eval analysis {} *****".format(args.my_task))
        for key, value in analysis.items():
            print("  {} = {}".format(key, value))
