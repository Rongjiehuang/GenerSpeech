# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import BaseWrapperDataset
from . import data_utils


class AddTargetDatasetSid(BaseWrapperDataset):
    def __init__(self, dataset, labels, pad, eos, batch_targets, process_label=None, add_to_input=False):
        super().__init__(dataset)
        self.labels = labels
        self.batch_targets = batch_targets
        self.pad = pad
        self.eos = eos
        self.process_label = process_label
        self.add_to_input = add_to_input

    def get_label(self, index):
        return self.labels[index] if self.process_label is None else self.process_label(self.labels[index])

    def __getitem__(self, index):
        item = self.dataset[index]
        item["label"] = self.get_label(index)
        item["label_cache"] = self.get_label(index)
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = len(self.get_label(index))
        return (sz, own_sz)

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        target = [s["label"] for s in samples if s["id"] in indices]

        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
            target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)

        collated["target"] = target

        if self.add_to_input:
            eos = target.new_full((target.size(0), 1), self.eos)
            collated["target"] = torch.cat([target, eos], dim=-1).long()
            collated["net_input"]["prev_output_tokens"] = torch.cat([eos, target], dim=-1).long()
        return collated




class AddTargetDatasetMulti(BaseWrapperDataset):
    def __init__(self, dataset, labels, pad, eos, batch_targets, process_label=None, add_to_input=False):
        super().__init__(dataset)
        self.labels_spk, self.labels_emo = labels
        self.batch_targets = batch_targets
        self.pad = pad
        self.eos = eos
        self.spk_label, self.emo_label = process_label
        self.add_to_input = add_to_input

    def get_spk_label(self, index):
        return self.labels_spk[index] if self.spk_label is None else self.spk_label(self.labels_spk[index])

    def get_emo_label(self, index):
        return self.labels_emo[index] if self.emo_label is None else self.emo_label(self.labels_emo[index])

    def __getitem__(self, index):
        item = self.dataset[index]
        item["label_spk"] = self.get_spk_label(index)
        item["label_emo"] = self.get_emo_label(index)
        item["label_spk_cache"] = self.get_spk_label(index)
        item["label_emo_cache"] = self.get_emo_label(index)
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = len(self.get_spk_label(index))
        return (sz, own_sz)

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        target_spk = [s["label_spk"] for s in samples if s["id"] in indices]
        target_emo = [s["label_emo"] for s in samples if s["id"] in indices]

        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in target_spk])
            target_spk = data_utils.collate_tokens(target_spk, pad_idx=self.pad, left_pad=False)
            target_emo = data_utils.collate_tokens(target_emo, pad_idx=self.pad, left_pad=False)

        collated["target_emo"] = target_emo
        collated["target_spk"] = target_spk

        return collated
