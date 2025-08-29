import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from .matching_layer import MatchingLayer
from transformers.models.t5.modeling_t5 import T5LayerNorm
from einops import rearrange

import torch.nn.functional as F

import os

class Model(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.inference = InferenceLayer(config)
        self.matching = MatchingLayer(config)

        self.dropout = torch.nn.Dropout(0.1)

        self.cls_edge = nn.Linear(768, 768)
        self.cls_senti = nn.Linear(768, 768)

        self.table_senti = build_table()
        self.table_edge = build_table()

        self.table1 = nn.Linear(768 * 2, 768)
        self.table2 = nn.Linear(768 * 2, 768)

    def forward(self, input_ids, attention_mask, ids, length, labels_edge, ID, labels_senti, table_labels_S=None, table_labels_E=None, pairs_true=None,):
        seq = self.bert(input_ids, attention_mask)[0]

        table_edge = self.table_edge(seq, length)

        table_senti = self.table_senti(seq, length)

        epoch = self.config.current_epoch
        output = self.inference(table_edge, attention_mask, table_labels_S, table_labels_E, labels_edge, labels_senti, table_senti, ID, epoch)
        output['ids'] = ids
        output = self.matching(output, table_senti, pairs_true, table_edge)
        return output


class InferenceLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cls_linear_S = nn.Linear(768, 1)
        self.cls_linear_E = nn.Linear(768, 1)
        self.proxy_edge = Proxy_Anchor(4, 768, 0.05,  32)
        self.proxy_senti = Proxy_Anchor(4, 768, 0.05, 32)

    def span_pruning(self, pred, z, attention_mask):
        mask_length = attention_mask.sum(dim=1) - 2
        length = ((attention_mask.sum(dim=1) - 2) * z).long()
        length[length < 10] = 10
        max_length = mask_length ** 2
        for i in range(length.shape[0]):
            if length[i] > max_length[i]:
                length[i] = max_length[i]
        batch_size = attention_mask.shape[0]
        pred_sort, _ = pred.view(batch_size, -1).sort(descending=True)
        batchs = torch.arange(batch_size).to('cuda')
        topkth = pred_sort[batchs, length - 1].unsqueeze(1)
        return pred >= (topkth.view(batch_size, 1, 1))

    def forward(self, table_edge, attention_mask, table_labels_S, table_labels_E, labels_edge, labels_senti, table_senti, ID, epoch):
        outputs = {}

        loss = 0
        for i in range(table_edge.size(0)):
            table_cl_con = table_edge[i]
            labels_con = labels_edge[i]
            len = attention_mask[i].sum(dim=0)
            table_cl_con = table_cl_con[1:len - 1, 1:len - 1, :].contiguous()
            labels_con = labels_con[1: len - 1, 1: len - 1].contiguous()
            table_cl_con = table_cl_con.view(-1, 768)
            labels_con = labels_con.view(-1, 1).float()
            sim_loss_edge = 0.05 * self.proxy_edge(table_cl_con, labels_con)
            loss += sim_loss_edge
        outputs['CL_edge'] = loss / table_edge.size(0)

        loss = 0
        for i in range(table_senti.size(0)):
            table_cl_con = table_senti[i]
            labels_con = labels_senti[i]
            len = attention_mask[i].sum(dim=0)
            table_cl_con = table_cl_con[1:len - 1, 1:len - 1, :].contiguous()
            labels_con = labels_con[1: len - 1, 1: len - 1].contiguous()
            table_cl_con = table_cl_con.view(-1, 768)
            labels_con = labels_con.view(-1, 1).float()
            sim_loss_edge = 0.05 * self.proxy_senti(table_cl_con, labels_con)
            loss += sim_loss_edge
        outputs['CL_senti'] = loss / table_senti.size(0)

        logits_S = torch.squeeze(self.cls_linear_S(table_edge), 3)
        logits_E = torch.squeeze(self.cls_linear_E(table_edge), 3)
        loss_func1 = nn.BCEWithLogitsLoss(weight=(table_labels_S >= 0))


        outputs['table_loss_S'] = loss_func1(logits_S, table_labels_S.float())
        outputs['table_loss_E'] = loss_func1(logits_E, table_labels_E.float())

        S_pred = torch.sigmoid(logits_S) * (table_labels_S >= 0)
        E_pred = torch.sigmoid(logits_E) * (table_labels_S >= 0)

        if self.config.span_pruning != 0:
            table_predict_S = self.span_pruning(S_pred, self.config.span_pruning, attention_mask)
            table_predict_E = self.span_pruning(E_pred, self.config.span_pruning, attention_mask)
        else:
            table_predict_S = S_pred > 0.5
            table_predict_E = E_pred > 0.5

        outputs['table_predict_S'] = table_predict_S
        outputs['table_predict_E'] = table_predict_E
        outputs['table_labels_S'] = table_labels_S
        outputs['table_labels_E'] = table_labels_E
        return outputs

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg, alpha):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(F.normalize(X,dim=1), F.normalize(P,dim=1))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss

class build_table(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv_1 = nn.Conv2d(
            in_channels=768,
            out_channels=768,
            kernel_size=(1, 1),
            padding=0
        )
        self.conv_2 = nn.Conv2d(
            in_channels=768,
            out_channels=768,
            kernel_size=(3, 3),
            padding=1
        )
        self.conv_3 = nn.Conv2d(
            in_channels=768,
            out_channels=768,
            kernel_size=(1, 1),
            padding=0
        )
        self.conv_4 = nn.Conv2d(
            in_channels=768,
            out_channels=768,
            kernel_size=(3, 3),
            padding=1
        )
        self.conv_5 = nn.Conv2d(
            in_channels=768,
            out_channels=768,
            kernel_size=(1, 1),
            padding=0
        )
        self.norm1 = T5LayerNorm(768, 1e-12)
        self.norm2 = T5LayerNorm(768, 1e-12)
        self.norm3 = T5LayerNorm(768, 1e-12)
        self.norm4 = T5LayerNorm(768, 1e-12)
        self.norm5 = T5LayerNorm(768, 1e-12)
        self.table = nn.Linear(768 * 2, 768)
        self.gelu = torch.nn.GELU()

    def forward(self, seq, length):
        seq_table = seq.unsqueeze(2).expand([-1, -1, length + 1, -1])
        seq_table_T = seq_table.transpose(1, 2)
        table = torch.cat((seq_table, seq_table_T), dim=3)
        table = self.table(table)
        table = torch.relu(table)
        D1 = rearrange(table, 'b m n d -> b d m n')
        n = D1.size(-1)

        D1_4 = self.conv_1(D1)
        D1_4 = rearrange(D1_4, 'b d m n-> b (m n) d ', n=n)
        D1_4 = self.norm1(D1_4)
        D1_4 = torch.relu(D1_4)
        D1_4 = rearrange(D1_4, 'b (m n) d -> b d m n', n=n)

        D1_1 = self.conv_2(D1_4)
        D1_1 = rearrange(D1_1, 'b d m n -> b (m n) d', n=n)
        D1_1 = self.norm2(D1_1)
        D1_1 = torch.relu(D1_1)
        D1_1 = rearrange(D1_1, 'b (m n) d -> b d m n', n=n)

        D1_2 = self.conv_3(D1_1)
        D1_2 = rearrange(D1_2, 'b d m n-> b (m n) d ', n=n)
        D1_2 = self.norm3(D1_2)
        D1_2 = torch.relu(D1_2)
        D1_2 = rearrange(D1_2, 'b (m n) d -> b d m n', n=n)

        D1_3 = self.conv_4(D1_2)
        D1_3 = rearrange(D1_3, 'b d m n-> b (m n) d ', n=n)
        D1_3 = self.norm4(D1_3)
        D1_3 = torch.relu(D1_3)
        D1_3 = rearrange(D1_3, 'b (m n) d -> b m n d', n=n)

        D1_4 = rearrange(D1_4, 'b d m n-> b m n d ', n=n)

        table = D1_4 + D1_3

        table = rearrange(table, 'b m n d-> b d m n', n=n)
        table = self.conv_5(table)
        table = rearrange(table, 'b d m n-> b m n d', n=n)

        return table