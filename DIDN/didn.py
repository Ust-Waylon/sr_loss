import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import numpy as np
import pandas as pd
from scipy import stats
import sys


class DIDN(nn.Module):
    """Dynamic intent-aware Iterative Denoising Network Class

    Args:
        n_items(int): the number of items
        hidden_size(int): the hidden size of gru
        position_embed_dim/embedding_dim(int): the dimension of item embedding
        batch_size(int):
        n_layers(int): the number of gru layers

    """

    def __init__(self, n_items, hidden_size, embedding_dim, batch_size, max_len, position_embed_dim, alpha1,  alpha2, alpha3, pos_num,neighbor_num, n_softmaxes=1, n_layers=1):
        super(DIDN, self).__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.position_embed_dim = position_embed_dim
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.pos_num = pos_num
        self.neighbor_num = neighbor_num
        self.n_softmaxes = n_softmaxes
        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)

        self.position_emb = nn.Embedding(self.pos_num, self.position_embed_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.3)
        self.position_dropout = nn.Dropout(0.3)
        self.dropout15 = nn.Dropout(0.15)
        self.dropout30 = nn.Dropout(0.30)
        self.dropout40 = nn.Dropout(0.40)
        self.dropout70 = nn.Dropout(0.70)

        # batchnormalization
        self.bn = torch.nn.BatchNorm1d(max_len, affine=False)
        self.bn1 = torch.nn.BatchNorm1d(embedding_dim, affine=False)
        self.final2std_cur = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.final2std_last = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.v_f2std = nn.Linear(self.embedding_dim, 1, bias=True)

        self.final2std2_std = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.final2std2_cur = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.v_f2std2 = nn.Linear(self.embedding_dim, 1, bias=True)

        self.final2std3_std = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.final2std3_cur = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.v_f2std3 = nn.Linear(self.embedding_dim, 1, bias=True)

        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)

        self.user2item_dim = nn.Linear(self.hidden_size, self.embedding_dim, bias=True)
        self.pos2item_dim = nn.Linear(self.position_embed_dim, self.embedding_dim, bias=True)
        # Dual gating mechanism
        self.w_u_z = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.w_u_r = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.w_u = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        self.u_u_z = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.u_u_r = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.u_u = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        self.w_p_f = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.w_p_i = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.w_p_c = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        self.u_p_f = nn.Linear(self.position_embed_dim, self.embedding_dim, bias=True)
        self.u_p_i = nn.Linear(self.position_embed_dim, self.embedding_dim, bias=True)
        self.u_p_c = nn.Linear(self.position_embed_dim, self.embedding_dim, bias=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.v_n2se = nn.Linear(self.embedding_dim, 1, bias=True)


        self.neighbor_v_t = nn.Linear(self.embedding_dim, 1, bias=True)


        self.merge_n_c = nn.Linear(self.embedding_dim*4, self.embedding_dim, bias=True)
        # attention to get stand
        self.v1_w = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.v1 = nn.Linear(self.embedding_dim, 1, bias=True)

        self.b = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.sf = nn.Softmax()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.layernorm_sess = nn.LayerNorm(self.embedding_dim)
        self.layernorm_item = nn.LayerNorm(self.embedding_dim)

        # Mixture of Softmaxes
        self.prior_vectors = nn.Embedding(self.n_softmaxes, self.embedding_dim)
        self.linear_layers = nn.Linear(self.embedding_dim, self.embedding_dim * self.n_softmaxes)
        self.tanh_mos = nn.Tanh()

    def forward(self, seq, lengths, normalize_emb = False):
        seq = seq.transpose(0, 1) # added to make the code compatible with the dataloader

        # get original item embedding
        embs_origin = self.emb_dropout(self.emb(seq))
        lengths_ori = lengths
        embs_origin = embs_origin.permute(1, 0, 2)

        # get position embedding  pos+length
        item_position = torch.tensor(range(1, seq.size()[0]+1), device=self.device)
        item_position = item_position.unsqueeze(1).expand_as(seq).permute(1, 0)
        len_d = torch.Tensor(lengths_ori).unsqueeze(1).expand_as(seq.permute(1, 0))*5 - 1
        len_d = len_d.type(torch.cuda.LongTensor)
        mask_position = torch.where(seq.permute(1, 0) > 0, torch.tensor([1], device=self.device),
                           torch.tensor([0], device=self.device))
        item_position = item_position*mask_position*len_d

        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device=self.device),
                           torch.tensor([0.], device=self.device))

        hidden = self.init_hidden(seq.size(1))
        # embs_padded = pack_padded_sequence(embs_origin.permute(1, 0, 2), lengths)
        embs_padded = pack_padded_sequence(embs_origin.permute(1, 0, 2), lengths, enforce_sorted=False) # added to make the code compatible with the dataloader
        gru_out, hidden = self.gru(embs_padded, hidden)
        # get user embeding gru_out (19,512,100)
        gru_out, lengths = pad_packed_sequence(gru_out)
        ht = hidden[-1]

        pos_embs = self.position_dropout(self.position_emb(item_position))
        pos_embs = self.dropout30(self.pos2item_dim(pos_embs))
        user_embs = self.dropout30(self.user2item_dim(ht))
        user_emb_expand = user_embs.unsqueeze(1).expand_as(embs_origin)

        user_z = torch.sigmoid(self.w_u_z(user_emb_expand)+self.u_u_z(embs_origin))
        user_r = torch.sigmoid(self.w_u_r(user_emb_expand) + self.u_u_r(embs_origin))
        uw_emb_h = self.tanh(self.w_u(user_emb_expand) + self.u_u(user_r*embs_origin))
        uw_emb = (1-user_z)*embs_origin + user_z * uw_emb_h

        pos_f = torch.sigmoid(self.w_p_f(uw_emb) + self.u_p_f(pos_embs))
        pos_i = torch.sigmoid(self.w_p_i(uw_emb) + self.u_p_i(pos_embs))
        pw_emb_h = self.tanh(self.w_p_c(uw_emb) + self.u_p_c(pos_embs))

        embs_final = pos_f*embs_origin + pos_i*pw_emb_h

        # batchnormalization
        embs_final = self.bn(embs_final)

        qq_current = self.final2std_cur(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())
        v_atten_q = self.v1_w(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())
        v_atten_q = self.tanh(v_atten_q)
        v_atten_w = self.v1(v_atten_q).expand_as(qq_current) * qq_current
        qq_avg_masked = mask.unsqueeze(2).expand_as(qq_current) * v_atten_w
        qq_avg_masked = torch.sum(qq_avg_masked, 1)
        qq_avg_masked = mask.unsqueeze(2).expand_as(qq_current) * qq_avg_masked.unsqueeze(1).expand_as(qq_current)

        beta1 = self.v_f2std(torch.sigmoid(qq_current + qq_avg_masked ).view(-1, self.embedding_dim)).view(
            mask.size())

        beta1 = self.sf(beta1)
        beta1_v = torch.mean(beta1, 1, True)[0].expand_as(beta1)
        beta1_mask = beta1 - self.alpha1 * beta1_v

        beta1 = torch.where(beta1_mask > 0, beta1,
                           torch.tensor([0.], device=self.device))

        sess_std = torch.sum(beta1.unsqueeze(2).expand_as(embs_final) * embs_final, 1)

        sess_std = self.dropout70(sess_std)

        q22_current = self.final2std2_cur(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())

        q22_std = self.final2std2_std(sess_std)
        q22_std_expand = q22_std.unsqueeze(1).expand_as(q22_current)
        q22_std_masked = mask.unsqueeze(2).expand_as(q22_current) * q22_std_expand

        beta2 = self.v_f2std2(torch.sigmoid(q22_current + q22_std_masked ).view(-1, self.embedding_dim)).view(mask.size())

        beta2 = self.sf(beta2)
        beta2_v = torch.mean(beta2, 1, True)[0].expand_as(beta2)
        beta2_mask = beta2 - self.alpha2 * beta2_v

        beta2 = torch.where(beta2_mask > 0, beta2,
                            torch.tensor([0.], device=self.device))
        sess_std2 = torch.sum(beta2.unsqueeze(2).expand_as(embs_final) * embs_final, 1)
        sess_std2 = self.dropout30(sess_std2)

        q3_current = self.final2std3_cur(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())

        q3_std = self.final2std3_std(sess_std2)
        q3_std_expand = q3_std.unsqueeze(1).expand_as(q3_current)
        q3_std_masked = mask.unsqueeze(2).expand_as(q3_current) * q3_std_expand

        beta3 = self.v_f2std3(torch.sigmoid(q3_current + q3_std_masked ).view(-1, self.embedding_dim)).view(mask.size())

        beta3 = self.sf(beta3)
        beta3_v = torch.mean(beta3, 1, True)[0].expand_as(beta3)
        beta3_mask = beta3 - self.alpha3 * beta3_v

        beta3 = torch.where(beta3_mask > 0, beta3,
                            torch.tensor([0.], device=self.device))
        sess_std3 = torch.sum(beta3.unsqueeze(2).expand_as(embs_final) * embs_final, 1)

        sess_current = sess_std3

        # cosine similarity
        fenzi = torch.matmul(sess_current, sess_current.permute(1, 0)) #512*512
        fenmu_l = torch.sum(sess_current * sess_current + 0.000001, 1)
        fenmu_l = torch.sqrt(fenmu_l).unsqueeze(1)
        fenmu = torch.matmul(fenmu_l, fenmu_l.permute(1, 0))
        cos_sim = fenzi / fenmu #512*512
        cos_sim = nn.Softmax(dim=-1)(cos_sim)

        k_v = self.neighbor_num
        if cos_sim.size()[0] < k_v:
            k_v = cos_sim.size()[0]
        cos_topk, topk_indice = torch.topk(cos_sim, k=k_v, dim=1)
        cos_topk = nn.Softmax(dim=-1)(cos_topk)
        sess_topk = sess_current[topk_indice]

        cos_sim = cos_topk.unsqueeze(2).expand(cos_topk.size()[0], cos_topk.size()[1], self.embedding_dim)

        neighbor_sess = torch.sum(cos_sim * sess_topk, 1)
        neighbor_sess = self.dropout40(neighbor_sess)
        sess_final = torch.cat(
            [sess_current, neighbor_sess, sess_current + neighbor_sess, sess_current * neighbor_sess], 1)
        
        # RF_result = self.NDF(sess_final)

        sess_final = self.dropout30(sess_final)
        sess_final = self.merge_n_c(sess_final) # [B, embedding_dim]

        item_embs = self.emb(torch.arange(self.n_items).to(self.device))
        item_embs = self.dropout15(item_embs)

        item_embs = self.bn1(item_embs)
        item_embs = self.b(item_embs) # [n_items, embedding_dim]

        if normalize_emb:
            # calculate the cosine similarity between the item embeddings and the session embeddings
            item_embs = self.layernorm_item(item_embs)
            sess_final = self.layernorm_sess(sess_final)

        if self.n_softmaxes > 1:
            # project session embeddings
            projected_emb = self.linear_layers(sess_final) # [B, embedding_dim * n_softmaxes]
            projected_emb = projected_emb.view(projected_emb.size(0), self.n_softmaxes, self.embedding_dim) # [B, n_softmaxes, embedding_dim]
            projected_emb = self.tanh_mos(projected_emb)
            
            # calculate softmax probabilities
            logits = torch.matmul(projected_emb, item_embs.permute(1, 0)) # [B, n_softmaxes, n_items]
            softmax_probs = torch.softmax(logits, dim=-1) # [B, n_softmaxes, n_items]

            # calculate prior probabilities (dot product between session embeddings and prior vectors, followed by softmax)
            prior_vectors = self.prior_vectors(torch.arange(self.n_softmaxes).to(self.device)) # [n_softmaxes, embedding_dim]
            prior_logits = torch.matmul(sess_final, prior_vectors.permute(1, 0)) # [B, n_softmaxes]
            prior_probs = torch.softmax(prior_logits, dim=-1).unsqueeze(-1) # [B, n_softmaxes, 1]

            # weighted sum of softmax probabilities
            probs = torch.sum(softmax_probs * prior_probs, dim=1) # [B, n_items]

            # skip connection
            original_probs = torch.matmul(sess_final, item_embs.permute(1, 0)) # [B, n_items]
            original_probs = torch.softmax(original_probs, dim=-1) # [B, n_items]
            probs = (probs + original_probs) / 2

            # NOTE: when n_softmaxes > 1, the forward function returns the probabilities rather than the scores (logits)
            return probs

        scores = torch.matmul(sess_final, item_embs.permute(1, 0))

        if normalize_emb:
            scores = scores * 12

        # scores = (scores + RF_result) / 2
        # scores = torch.matmul(sess_final, item_embs.permute(1, 0))

        return scores

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def transpose_for_scores(self, x, attention_head_size):
        # INPUT:  x'shape = [bs, seqlen, hid_size] hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)  # [bs, 8, seqlen, 16]
