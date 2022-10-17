import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import json
from logging import getLogger
logger = getLogger()


class MatchPyramid(torch.nn.Module):

    def __init__(self, args, weights_matrix, pos_weights_matrix):
        super().__init__()
        self.args = args
        self.embeds_dim = args.word_dim
        self.embeds = nn.Embedding(args.word_vocab_size, args.word_dim)
        self.embeds.weight.data.copy_(weights_matrix)
        self.embeds.weight.requires_grad = False
        if args.use_pos_emb:
            self.pos_embeds = nn.Embedding(args.max_len, args.word_dim)
            self.pos_embeds.weight.data.copy_(pos_weights_matrix)
        self.max_len1 = args.max_len
        self.max_len2 = args.max_len
        self.conv1_size = [5, 5, 8]
        self.pool1_size = [10, 10]
        self.conv2_size = [3, 3, 16]
        self.pool2_size = [5, 5]
        self.dim_hidden = 128

        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=self.conv1_size[-1],
                                     kernel_size=tuple(
                                         self.conv1_size[0:2]),
                                     padding=0,
                                     bias=True
                                     )
        # torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1_size[-1],
                                     out_channels=self.conv2_size[-1],
                                     kernel_size=tuple(
                                         self.conv2_size[0:2]),
                                     padding=0,
                                     bias=True
                                     )
        self.pool1 = torch.nn.AdaptiveMaxPool2d(tuple(self.pool1_size))
        self.pool2 = torch.nn.AdaptiveMaxPool2d(tuple(self.pool2_size))
        self.linear1 = torch.nn.Linear(self.pool2_size[0] * self.pool2_size[1] * self.conv2_size[-1],
                                       self.dim_hidden, bias=True)
        # torch.nn.init.kaiming_normal_(self.linear1.weight)
        self.linear2 = torch.nn.Linear(self.dim_hidden, 2, bias=True)
        # torch.nn.init.kaiming_normal_(self.linear2.weight)
        if logger:
            self.logger = logger
            self.logger.info("Hyper Parameters of MatchPyramid: %s" % json.dumps(
                {"Kernel": [self.conv1_size, self.conv2_size],
                 "Pooling": [self.pool1_size, self.pool2_size],
                 "MLP": self.dim_hidden}))

    def forward(self, **kwargs):
        x1 = self.embeds(kwargs['text_a'])
        x2 = self.embeds(kwargs['text_b'])
        if self.args.use_pos_emb:
            # print("use pos emb!")
            pos_emb = self.pos_embeds(kwargs['pos_ids'])
            x1 += pos_emb / (kwargs['text_a_len']**0.25).view(x1.size()[0], 1, 1)
            x2 += pos_emb / (kwargs['text_b_len']**0.25).view(x2.size()[0], 1, 1)

        # x1,x2:[batch, seq_len, dim_xlm]
        bs, seq_len1, dim_xlm = x1.size()
        seq_len2 = x2.size()[1]
        pad1 = self.max_len1 - seq_len1
        pad2 = self.max_len2 - seq_len2
        # simi_img:[batch, 1, seq_len, seq_len]
        # x1_norm = x1.norm(dim=-1, keepdim=True)
        # x1_norm = x1_norm + 1e-8
        # x2_norm = x2.norm(dim=-1, keepdim=True)
        # x2_norm = x2_norm + 1e-8
        # x1 = x1 / x1_norm
        # x2 = x2 / x2_norm
        # use cosine similarity since dim is too big for dot-product
        simi_img = torch.matmul(x1, x2.transpose(1, 2)) / np.sqrt(dim_xlm)
        if pad1 != 0 or pad2 != 0:
            simi_img = F.pad(simi_img, (0, pad2, 0, pad1))
        assert simi_img.size() == (bs, self.max_len1, self.max_len2)
        simi_img = simi_img.unsqueeze(1)
        # self.logger.info(simi_img.size())
        # [batch, 1, conv1_w, conv1_h]
        simi_img = F.relu(self.conv1(simi_img))
        # [batch, 1, pool1_w, pool1_h]
        simi_img = self.pool1(simi_img)
        # [batch, 1, conv2_w, conv2_h]
        simi_img = F.relu(self.conv2(simi_img))
        # # [batch, 1, pool2_w, pool2_h]
        simi_img = self.pool2(simi_img)
        # assert simi_img.size()[1] == 1
        # [batch, pool1_w * pool1_h * conv2_out]
        simi_img = simi_img.squeeze(1).view(bs, -1)
        # output = self.linear1(simi_img)
        output = self.linear2(F.relu(self.linear1(simi_img)))
        return output

    def get_embeddings(self, sentence, pos_ids, sentence_lens):
        x = self.embeds(sentence)
        return x
