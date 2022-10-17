from torch import nn
import torch
import torch.nn.functional as F
from utils.model_utils import sort_by_seq_lens
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class ESIM(nn.Module):
    def __init__(self, args, weights_matrix, pos_weights_matrix):
        super(ESIM, self).__init__()
        self.args = args
        self.dropout = 0.5
        self.hidden_size = args.hidden_size
        self.embeds_dim = args.word_dim
        self.embeds = nn.Embedding(args.word_vocab_size, args.word_dim)
        self.embeds.weight.data.copy_(weights_matrix)
        self.embeds.weight.requires_grad = False
        if args.use_pos_emb:
            self.pos_embeds = nn.Embedding(args.max_len, args.hidden_size)
            self.pos_embeds.weight.data.copy_(pos_weights_matrix)
        self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)
        self.lstm1 = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_size*8, self.hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, args.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(args.linear_size, args.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(args.linear_size, 2),
            nn.Softmax(dim=-1)
        )
    
    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x_align: batch_size * seq_len * hidden_size

        return x1_align, x2_align

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def pad_encoder(self, sequences_batch, sequences_lengths):
        sorted_batch, sorted_lengths, _, restoration_idx = sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = pack_padded_sequence(sorted_batch, sorted_lengths, batch_first=True)
        outputs, _ = self.lstm1(packed_batch)
        outputs, _ = pad_packed_sequence(outputs, total_length=self.args.max_len, batch_first=True)
        reordered_outputs = outputs.index_select(0, restoration_idx)
        # print("second: ", reordered_outputs.size())
        return reordered_outputs

    def forward(self, **kwargs):
        # batch_size * seq_len
        sent1, sent2 = kwargs['text_a'], kwargs['text_b']
        sent1_lens, sent2_lens = kwargs['text_a_len'], kwargs['text_b_len']
        mask1, mask2 = sent1.eq(0), sent2.eq(0)

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.embeds(sent1)
        x2 = self.embeds(sent2)

        x1 = self.bn_embeds(x1.transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(x2.transpose(1, 2).contiguous()).transpose(1, 2)

        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1 = self.pad_encoder(x1, sent1_lens)
        o2 = self.pad_encoder(x2, sent2_lens)
        if self.args.use_pos_emb:
            # print("use pos emb!")
            pos_emb = self.pos_embeds(kwargs['pos_ids'])
            cat_pos_emb = torch.cat([pos_emb, torch.flip(pos_emb, dims=[1])], dim=2)
            o1 = o1 + cat_pos_emb / (kwargs['text_a_len']**0.5).view(x1.size()[0], 1, 1)
            o2 = o2 + cat_pos_emb / (kwargs['text_b_len']**0.5).view(x2.size()[0], 1, 1)

        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)
        
        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)

        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)
        return similarity#, o1, o2, x

    def get_embeddings(self, sentence, sentence_lens, pool):
        x = self.embeds(sentence)
        bn_x = self.bn_embeds(x.transpose(1, 2).contiguous()).transpose(1, 2)

        sorted_batch, sorted_lengths, _, restoration_idx = sort_by_seq_lens(bn_x, sentence_lens)
        packed_batch = pack_padded_sequence(sorted_batch, sorted_lengths, batch_first=True)
        h, (h_t, _) = self.lstm1(packed_batch)
        if pool == 'last':
            h_t = torch.cat((h_t[-1], h_t[-2]), 1)
        elif pool == 'max':
            h_tmp, _ = pad_packed_sequence(h, total_length=self.args.max_len, batch_first=True)
            h_t = torch.max(h_tmp, 1)[0].squeeze()
            # print("h_t shape: ", h_t.size())

        embeddings = h_t.detach().cpu().numpy()
        return embeddings

    def get_final_hidden(self, sent1, sent2, pos_ids, text_a_len, text_b_len):
        # batch_size * seq_len
        mask1, mask2 = sent1.eq(0), sent2.eq(0)

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.embeds(sent1)
        x2 = self.embeds(sent2)

        x1 = self.bn_embeds(x1.transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(x2.transpose(1, 2).contiguous()).transpose(1, 2)

        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)
        if self.args.use_pos_emb:
            # print("use pos emb!")
            pos_emb = self.pos_embeds(pos_ids)
            cat_pos_emb = torch.cat([pos_emb, torch.flip(pos_emb, dims=[1])], dim=2)
            o1 = o1 + cat_pos_emb / (text_a_len ** 0.5).view(x1.size()[0], 1, 1)
            o2 = o2 + cat_pos_emb / (text_b_len ** 0.5).view(x2.size()[0], 1, 1)

        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)

        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)

        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)

        return x