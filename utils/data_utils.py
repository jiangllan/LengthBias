import os
import pickle

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from torch.utils.data import Dataset


def read_tsv(input_file, col_num, seperated="\t", skip_header=True):
    columns = None
    with open(input_file) as f:
        lines = f.readlines()
    output = []
    for line in lines:
        case = line.strip().split(seperated)
        if len(case) == col_num:
            output.append(case)
    if skip_header:
        columns = output[0]
        data = np.array(output[1:])
    else:
        data = np.array(output)
    return columns, data


def read_dataset(args, split):
    file_path = os.path.join(args.data_dir, args.my_task, args.delta_lens_mode, args.balanced_mode, "{}.tsv".format(split))
    print("read dataset from file: ", file_path)
    if args.my_task.lower() == "qqp":
        if args.balanced_mode and args.delta_lens_mode:
            data = pd.read_csv(file_path, sep="\t")
            labels = None if split == "test" else data["label"].tolist()
            dataset = {"text_a": data["sentence1"].tolist(), "text_b": data["sentence2"].tolist(), "label": labels}
        else:
            _, data = read_tsv(file_path, col_num=6)
            dataset = {"text_a": data[:, 3].tolist(), "text_b": data[:, 4].tolist(), "label": data[:, 5].tolist()}
        return dataset
    else:
        data = pd.read_csv(file_path, sep="\t")
        return {"text_a": data["sentence1"].tolist(), "text_b": data["sentence2"].tolist(),
                "label": data["label"].tolist()}

def build_embedding_matrix(args, tok, max_words):
    # build vocabulary
    print("building vocabulary...")
    f = open(os.path.join(args.embedding_dir, "glove.840B.300d.txt"))
    embeddings_index = {}
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        if word in tok.word_index:
            embeddings_index[word] = coefs
    f.close()
    print("building vocabulary over.")

    # build weights_matrix
    weights_matrix = np.zeros((max_words, 300))
    for word, i in tok.word_index.items():
        try:
            weights_matrix[i] = embeddings_index[word]
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.1, size=(300,))
    return weights_matrix


def load_embedding_matrix(args, tok, max_words):
    cache_dir = os.path.join(args.data_dir, args.my_task)
    aux_file = os.path.join(cache_dir, "{}_weights_matrix.pickle".format(args.my_task.lower()))
    if not os.path.exists(aux_file):
        print("building weights matrix from raw data...")
        weights_matrix = build_embedding_matrix(args, tok, max_words)
        with open(aux_file, "wb") as f:
            pickle.dump(weights_matrix, f)
    else:
        print("load weights matrix from cached file: ", aux_file)
        with open(aux_file, "rb") as f:
            weights_matrix = pickle.load(f)
        print("load over.")
    return weights_matrix


def characterize(batch, characterized_words):
    """
    :param batch: Pytorch Variable with shape (batch, seq_len)
    :return: Pytorch Variable with shape (batch, seq_len, max_word_len)
    """
    batch = batch.data.cpu().numpy().astype(int).tolist()
    return [[characterized_words[w] for w in words] for words in batch]


def build_char_vocab(tok):
    char_vocab = {'': 0}
    max_word_len = max([len(w) for w in tok.word_index.keys()])
    characterized_words = [[0] * max_word_len, [0] * max_word_len]
    # for normal words
    for word in tok.word_index.keys():
        chars = []
        for c in list(word):
            if c not in char_vocab:
                char_vocab[c] = len(char_vocab)

            chars.append(char_vocab[c])

        chars.extend([0] * (max_word_len - len(word)))
        characterized_words.append(chars)
    return char_vocab, characterized_words


def build_word_vocab(args, train_data, dev_data):
    tok = Tokenizer()
    tok.fit_on_texts(train_data['text_a'] + train_data['text_b']
                     + dev_data['text_a'] + dev_data['text_b'])
    max_words = min(args.word_vocab_size, len(tok.word_index))
    tok.word_index = {e: i - 1 for e, i in tok.word_index.items() if
                      i <= max_words}  # <= because tokenizer is 1 indexed
    tok.word_index['UNK'] = max_words
    tok.word_index['PAD'] = max_words + 1
    return tok


def pad_sequences(args, sequences, dtype='int32', padding='post', truncating='post', value=0.):
    """ pad_sequences
    Arguments:
        sequences: 序列
        self.args.max_len: int 最大长度
        dtype: 转变后的数据类型
        padding: 填充方法'pre' or 'post'
        truncating: 截取方法'pre' or 'post'
        value: float 填充的值
    Returns:
        x: numpy array 填充后的序列维度为 (number_of_sequences, self.args.max_len)
    """
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if args.max_len < 0:
        args.max_len = np.max(lengths)
    x = (np.ones((nb_samples, args.max_len)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-args.max_len:]
        elif truncating == 'post':
            trunc = s[:args.max_len]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def sentence2index(tok, data, args):
    text_a_list, text_b_list = [item.split(' ') for item in data['text_a']], [item.split(' ') for item in data[
        "text_b"]]
    tokenized_text_a_list, tokenized_text_b_list = [], []
    text_a_length, text_b_length = [], []
    for text_a, text_b in zip(text_a_list, text_b_list):
        a = [tok.word_index[word] if word in tok.word_index.keys() else tok.word_index['UNK'] for
             word in text_a]
        b = [tok.word_index[word] if word in tok.word_index.keys() else tok.word_index['UNK'] for
             word in text_b]
        tokenized_text_a_list.append(a)
        text_a_length.append(min(len(a), args.max_len))
        tokenized_text_b_list.append(b)
        text_b_length.append(min(len(b), args.max_len))
    tokenized_text_a_list = pad_sequences(args, tokenized_text_a_list).tolist()
    tokenized_text_b_list = pad_sequences(args, tokenized_text_b_list).tolist()
    return tokenized_text_a_list, tokenized_text_b_list, np.array(text_a_length), np.array(text_b_length)


def preprocess(args):
    # read all text data
    train, dev = read_dataset(args, "train"), read_dataset(args, "dev")
    # build word vocab
    tok = build_word_vocab(args, train, dev)
    # build char vocab
    char_vocab, characterized_words = None, None
    if args.use_char_emb:
        char_vocab, characterized_words = build_char_vocab(tok)
    return tok, char_vocab, characterized_words


def tokenize(tok, args):
    # read all text data
    train, dev = read_dataset(args, "train"), read_dataset(args, "dev")
    # tokenize raw text
    train['tokenized_text_a'], train['tokenized_text_b'], \
    train['text_a_length'], train['text_b_length'] = sentence2index(tok, train, args)
    dev['tokenized_text_a'], dev['tokenized_text_b'], \
    dev['text_a_length'], dev['text_b_length'] = sentence2index(tok, dev, args)
    return train, dev


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding


class BaselineDataset(Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split
        self.tok, self.dataset, self.char_vocab, self.characterized_words = self.load_data()
        self.label_list = ['0', '1']
        # build vocabulary
        self.weights_matrix = load_embedding_matrix(args, self.tok, self.max_words)
        self.max_word_len = max([len(w) for w in self.tok.word_index.keys()])

    def load_data(self):
        # load/build tok, vocab
        cache_dir = os.path.join(self.args.data_dir, self.args.my_task)
        if os.path.exists(os.path.join(cache_dir, "char_vocab.pickle")):
            # load tok, vocab
            print("load aux tok file form: {}".format(os.path.join(cache_dir, "tok.pickle")))
            with open(os.path.join(cache_dir, "char_vocab.pickle"), "rb") as f:
                char_vocab = pickle.load(f)
            with open(os.path.join(cache_dir, "characterized_words.pickle"), "rb") as f:
                characterized_words = pickle.load(f)
            with open(os.path.join(cache_dir, "tok.pickle"), "rb") as f:
                tok = pickle.load(f)
        else:
            # build tok, vocab
            tok, char_vocab, characterized_words = preprocess(self.args)
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            with open(os.path.join(cache_dir, "char_vocab.pickle"), "wb") as f:
                pickle.dump(char_vocab, f)
            with open(os.path.join(cache_dir, "characterized_words.pickle"), "wb") as f:
                pickle.dump(characterized_words, f)
            with open(os.path.join(cache_dir, "tok.pickle"), "wb") as f:
                pickle.dump(tok, f)

        self.max_words = len(tok.word_index)
        print("dataset max words: ", self.max_words)

        data_cache_dir = os.path.join(self.args.data_dir, self.args.my_task, self.args.delta_lens_mode,
                                      self.args.balanced_mode)
        print("data cache dir: ", data_cache_dir)
        if os.path.exists(os.path.join(data_cache_dir, "{}.pickle".format(self.split))):
            print("load dataset from pickle files: ", os.path.join(data_cache_dir, "{}.pickle".format(self.split)))
            with open(os.path.join(data_cache_dir, "{}.pickle".format(self.split)), "rb") as f:
                dataset = pickle.load(f)
        else:
            print("build dataset.")
            train_dataset, dev_dataset = tokenize(tok, self.args)
            dataset = train_dataset if self.split == "train" else dev_dataset
            with open(os.path.join(data_cache_dir, "{}.pickle".format(self.split)), "wb") as f:
                pickle.dump(dataset, f)
        return tok, dataset, char_vocab, characterized_words

    def get_labels(self):
        return self.label_list

    def __len__(self):
        return len(self.dataset['text_a'])

    def __getitem__(self, index):
        text_a = self.dataset['tokenized_text_a'][index]
        text_a_len = self.dataset['text_a_length'][index]
        pos_ids = np.arange(self.args.max_len)
        text_b = self.dataset['tokenized_text_b'][index]
        text_b_len = self.dataset['text_b_length'][index]
        label = int(self.dataset['label'][index])
        return [text_a, text_b, pos_ids, text_a_len, text_b_len, label]
