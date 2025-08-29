from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy
import os
from . import load_json
import spacy
import json
import torch
import stanza
# from torch_geometric.data import Data
import math
import nlpaug.augmenter.word as naw
# stanza.download('en')
# 初始化同义词替换增强器
# aug = naw.SynonymAug(aug_src='wordnet')

polarity_map = {
    'NEG': 0,
    'NEU': 1,
    'POS': 2
}

polarity_map_reversed = {
    0: 'NEG',
    1: 'NEU',
    2: 'POS'
}


class Example:
    def __init__(self, data, max_length=-1, dep_map_path="../dep_map.json"):
        self.data = data
        self.max_length = max_length
        self.data['tokens'] = eval(str(self.data['tokens']))

        print()
        # self.parser = spacy.load('en_core_web_sm')

    def __getitem__(self, key):
        return self.data[key]

    def t_entities(self):
        return [tuple(entity[:3]) for entity in self['entities'] if entity[0] == 'target']

    def o_entities(self):
        return [tuple(entity[:3]) for entity in self['entities'] if entity[0] == 'opinion']

    def entity_label(self, target_oriented, length):
        entities = self.t_entities() if target_oriented else self.o_entities()
        return Example.make_start_end_labels(entities, length)

    def table_label(self, length, ty, id_len):
        label = [[-1 for _ in range(length)] for _ in range(length)]
        id_len = id_len.item()

        for i in range(1, id_len - 1):
            for j in range(1, id_len - 1):
                # if i != j:
                    label[i][j] = 0

        for t_start, t_end, o_start, o_end, pol in self['pairs']:
            if t_start - t_end >= 20 or o_start - o_end >= 20:
                print()
            if t_start != -1:
                if ty == 'S':
                    label[t_start + 1][o_start + 1] = 1
                elif ty == 'E':
                    label[t_end][o_end] = 1
            else:
                if ty == 'S':
                    label[o_start + 1][o_start + 1] = 1
                elif ty == 'E':
                    label[o_end][o_end] = 1
        return label

    def table_label_ia(self, length, ty, id_len):
        label = [[-1 for _ in range(length)] for _ in range(length)]
        id_len = id_len.item()

        for i in range(1, id_len - 1):
            for j in range(1, id_len - 1):
                if i == j:
                    label[i][j] = 0

        for t_start, t_end, o_start, o_end, pol in self['pairs']:
            if t_start == -1:
                if ty == 'S':
                    label[o_start + 1][o_start + 1] = 1
                elif ty == 'E':
                    label[o_end][o_end] = 1

        return label

    def label_edge(self, length, id_len):
        label = [[-1 for _ in range(length)] for _ in range(length)]
        id_len = id_len.item()

        for i in range(1, id_len - 1):
            for j in range(1, id_len - 1):
                # if i != j:
                label[i][j] = 0

        for t_start, t_end, o_start, o_end, pol in self['pairs']:
            if t_start != -1:
                if t_start + 1 !=t_end and o_start + 1 != o_end :
                    label[t_start + 1][o_start + 1] = 1
                    label[t_end][o_end] = 2
                else:
                    label[t_start + 1][o_start + 1] = 3
            else:
                if o_start + 1 != o_end:
                    label[o_start + 1][o_start + 1] = 1
                    label[o_end][o_end] = 2
                else:
                    label[o_start + 1][o_start + 1] = 3

        # label = [-1 for _ in range(length)]
        # id_len = id_len.item()
        #
        # for i in range(1, id_len - 1):
        #     label[i] = 0
        #
        # for t_start, t_end, o_start, o_end, pol in self['pairs']:
        #     if t_start != -1:
        #         for i in range(o_start + 1, o_end + 1):
        #             label[i] = 1
        #         for i in range(t_start + 1, t_end + 1):
        #             label[i] = 1
        #     else:
        #         for i in range(o_start + 1, o_end + 1):
        #             label[i] = 1
        return label

    def label_senti(self, length, id_len):
        label = [[-1 for _ in range(length)] for _ in range(length)]
        id_len = id_len.item()

        for i in range(1, id_len - 1):
            for j in range(1, id_len - 1):
                # if i != j:
                label[i][j] = 0

        for t_start, t_end, o_start, o_end, pol in self['pairs']:
            if t_start == -1:
                if pol == 'POS':
                    label[o_start + 1][o_start + 1] = 1
                    label[o_end][o_end] = 1
                    # for i in range(o_start + 1, o_end + 1):
                    #     for j in range(o_start + 1, o_end + 1):
                    #         label[i][j] = 1
                if pol == 'NEU':
                    label[o_start + 1][o_start + 1] = 2
                    label[o_end][o_end] = 2
                    # for i in range(t_start + 1, o_end + 1):
                    #     for j in range(o_start + 1, o_end + 1):
                    #         label[i][j] = 2
                if pol == 'NEG':
                    label[o_start + 1][o_start + 1] = 3
                    label[o_end][o_end] = 3
                    # for i in range(t_start + 1, o_end + 1):
                    #     for j in range(o_start + 1, o_end + 1):
                    #         label[i][j] = 3
            else:
                if pol == 'POS':
                    label[t_start + 1][o_start + 1] = 1
                    label[t_end][o_end] = 1
                    # for i in range(t_start + 1, t_end + 1):
                    #     for j in range(o_start + 1, o_end + 1):
                    #         label[i][j] = 1
                if pol == 'NEU':
                    label[t_start + 1][o_start + 1] = 2
                    label[t_end][o_end] = 2
                    # for i in range(t_start + 1, t_end + 1):
                    #     for j in range(o_start + 1, o_end + 1):
                    #         label[i][j] = 2
                if pol == 'NEG':
                    label[t_start + 1][o_start + 1] = 3
                    label[t_end][o_end] = 3
                    # for i in range(t_start + 1, t_end + 1):
                    #     for j in range(o_start + 1, o_end + 1):
                    #         label[i][j] = 3

        # label = [-1 for _ in range(length)]
        # id_len = id_len.item()
        #
        # for i in range(1, id_len - 1):
        #     label[i] = 0
        #
        # for t_start, t_end, o_start, o_end, pol in self['pairs']:
        #     if t_start != -1:
        #         if pol == 'POS':
        #             for i in range(o_start + 1, o_end + 1):
        #                 label[i] = 1
        #             for i in range(t_start + 1, t_end + 1):
        #                 label[i] = 1
        #         if pol == 'NEU':
        #             for i in range(o_start + 1, o_end + 1):
        #                 label[i] = 2
        #             for i in range(t_start + 1, t_end + 1):
        #                 label[i] = 2
        #         if pol == 'NEG':
        #             for i in range(o_start + 1, o_end + 1):
        #                 label[i] = 3
        #             for i in range(t_start + 1, t_end + 1):
        #                 label[i] = 3
        #     else:
        #         if pol == 'POS':
        #             for i in range(o_start + 1, o_end + 1):
        #                 label[i] = 1
        #         if pol == 'NEU':
        #             for i in range(o_start + 1, o_end + 1):
        #                 label[i] = 2
        #         if pol == 'NEG':
        #             for i in range(o_start + 1, o_end + 1):
        #                 label[i] = 3
        return label

    @staticmethod
    def make_start_end_labels(entities, length, plus_one=True):
        start_label = [0] * length
        end_label = [0] * length

        for (t, s, e) in entities:
            if plus_one:
                s, e = s + 1, e + 1

            if s < length:
                start_label[s] = 1

            if e - 1 < length:
                end_label[e - 1] = 1

        return start_label, end_label


class DataCollatorForASTE:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length


    def __call__(self, examples):

        batch = self.tokenizer_function(examples)

        length = batch['input_ids'].size(1)
        batch['t_start_labels'], batch['t_end_labels'] = self.start_end_labels(examples, True, length)
        batch['o_start_labels'], batch['o_end_labels'] = self.start_end_labels(examples, False, length)
        batch['example_ids'] = [example['ID'] for example in examples]
        batch['table_labels_S'] = torch.tensor(
            [examples[i].table_label(length, 'S', (batch['input_ids'][i] > 0).sum()) for i in range(len(examples))],
            dtype=torch.long)
        batch['table_labels_E'] = torch.tensor(
            [examples[i].table_label(length, 'E', (batch['input_ids'][i] > 0).sum()) for i in range(len(examples))],
            dtype=torch.long)
        # batch['table_labels_iaS'] = torch.tensor(
        #     [examples[i].table_label_ia(length, 'S', (batch['input_ids'][i] > 0).sum()) for i in
        #      range(len(examples))], dtype=torch.long)
        # batch['table_labels_iaE'] = torch.tensor(
        #     [examples[i].table_label_ia(length, 'E', (batch['input_ids'][i] > 0).sum()) for i in
        #      range(len(examples))], dtype=torch.long)
        batch['labels_edge'] = torch.tensor(
            [examples[i].label_edge(length, (batch['input_ids'][i] > 0).sum()) for i in
             range(len(examples))], dtype=torch.long)
        batch['labels_senti'] = torch.tensor(
            [examples[i].label_senti(length, (batch['input_ids'][i] > 0).sum()) for i in
             range(len(examples))], dtype=torch.long)
        al = [example['pairs'] for example in examples]
        pairs_ret = []
        for pairs in al:
            pairs_chg = []
            for p in pairs:
                pairs_chg.append([p[0], p[1], p[2], p[3], polarity_map[p[4]] + 1])
            pairs_ret.append(pairs_chg)
        batch['pairs_true'] = pairs_ret

        # if len(examples[0]['pairs']) > 10:
        #     print('ID:')
        #     print(examples[0]['ID'])
        #     print('Length:')
        #     print(len(examples[0]['pairs']))
        #     print('Token:')
        #     print(len(examples[0]['tokens']))


        return {
            'ids': batch['example_ids'],
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'table_labels_S': batch['table_labels_S'],
            'table_labels_E': batch['table_labels_E'],
            # 'table_labels_iaS': batch['table_labels_iaS'],
            # 'table_labels_iaE': batch['table_labels_iaE'],
            'pairs_true': batch['pairs_true'],
            'length': batch['length'],
            'labels_edge' : batch['labels_edge'],
            'labels_senti': batch['labels_senti'],
            'ID': examples[0]['ID'],
        }

    def start_end_labels(self, examples, target_oriented, length):
        start_labels = []
        end_labels = []

        for example in examples:
            start_label, end_label = example.entity_label(target_oriented, length)
            start_labels.append(start_label)
            end_labels.append(end_label)

        start_labels = torch.tensor(start_labels, dtype=torch.long)
        end_labels = torch.tensor(end_labels, dtype=torch.long)

        return start_labels, end_labels

    def tokenizer_function(self, examples):
        text = [example['sentence'] for example in examples]
        text_temp = text.copy()
        kwargs = {
            'text': text,
            'return_tensors': 'pt'
        }

        if self.max_seq_length in (-1, 'longest'):
            kwargs['padding'] = True
        else:
            kwargs['padding'] = 'max_length'
            kwargs['max_length'] = self.max_seq_length
            kwargs['truncation'] = True
        kwargs_temp = kwargs.copy()
        kwargs_temp['text'] = text_temp
        batch_encodings = self.tokenizer(**kwargs_temp)
        length = batch_encodings['input_ids'].size(1) - 1

        start_label_masks = []
        end_label_masks = []

        for i in range(len(examples)):
            encoding = batch_encodings[i]
            word_ids = encoding.word_ids
            type_ids = encoding.type_ids

            start_label_mask = [(1 if type_ids[i] == 0 else 0) for i in range(length)]
            end_label_mask = [(1 if type_ids[i] == 0 else 0) for i in range(length)]

            for token_idx in range(length):
                current_word_idx = word_ids[token_idx]
                prev_word_idx = word_ids[token_idx - 1] if token_idx - 1 > 0 else None
                next_word_idx = word_ids[token_idx + 1] if token_idx + 1 < length else None

                if prev_word_idx is not None and current_word_idx == prev_word_idx:
                    start_label_mask[token_idx] = 0
                if next_word_idx is not None and current_word_idx == next_word_idx:
                    end_label_mask[token_idx] = 0

            start_label_masks.append(start_label_mask)
            end_label_masks.append(end_label_mask)

        all_token_range = []
        for batch in range(len(examples)):
            temp_words = text[batch].split(" ")
            token_start = 1
            temp_range = []
            for i, w, in enumerate(temp_words):
                token_end = token_start + len(self.tokenizer.encode(w, add_special_tokens=False))
                temp_range.append([token_start, token_end - 1])
                token_start = token_end
                all_token_range.append(temp_range)

        batch_encodings = dict(batch_encodings)
        batch_encodings['start_label_masks'] = torch.tensor(start_label_masks, dtype=torch.long)
        batch_encodings['end_label_masks'] = torch.tensor(end_label_masks, dtype=torch.long)
        batch_encodings['length'] = length
        return batch_encodings


class ASTEDataModule(pl.LightningDataModule):
    def __init__(self, args):

        super(ASTEDataModule, self).__init__()
        self.args = args

        self.model_name_or_path = args.model_name_or_path
        self.max_seq_length = args.max_seq_length if args.max_seq_length > 0 else 'longest'
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size

        self.data_dir = args.prefix + args.dataset
        self.num_workers = args.num_workers
        self.cuda_ids = args.cuda_ids

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

    def load_dataset(self):
        train_file_name = os.path.join(self.data_dir, 'train.json')
        dev_file_name = os.path.join(self.data_dir, 'dev.json')
        test_file_name = os.path.join(self.data_dir, 'test.json')

        if not os.path.exists(dev_file_name):
            dev_file_name = test_file_name

        train_examples = [Example(data, self.max_seq_length) for data in load_json(train_file_name)]
        dev_examples = [Example(data, self.max_seq_length) for data in load_json(dev_file_name)]
        test_examples = [Example(data, self.max_seq_length) for data in load_json(test_file_name)]

        self.raw_datasets = {
            'train': train_examples,
            'dev': dev_examples,
            'test': test_examples
        }

    def get_dataloader(self, mode, batch_size, shuffle):
        dataloader = DataLoader(
            dataset=self.raw_datasets[mode],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=DataCollatorForASTE(tokenizer=self.tokenizer,
                                           max_seq_length=self.max_seq_length),
            pin_memory=True,
            prefetch_factor=16
        )

        print(mode, len(dataloader))
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.eval_batch_size, shuffle=False)