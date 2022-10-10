#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Wav2letter decoders.
"""

from collections import namedtuple, deque
import gc
import itertools as it
import numpy as np
import torch
torch.set_printoptions(profile="full")
import os.path as osp
import warnings
from fairseq import tasks
from fairseq.utils import apply_to_sample
from examples.speech_recognition.data.replabels import unpack_replabels

try:
    from wav2letter.common import create_word_dict, load_words
    from wav2letter.criterion import CpuViterbiPath, get_data_ptr_as_bytes
    from wav2letter.decoder import (
        CriterionType,
        DecoderOptions,
        KenLM,
        LM,
        LMState,
        SmearingMode,
        Trie,
        LexiconDecoder,
        LexiconFreeDecoder,
    )
except:
    warnings.warn(
        "wav2letter python bindings are required to use this functionality. Please install from https://github.com/facebookresearch/wav2letter/wiki/Python-bindings"
    )
    LM = object
    LMState = object



class LidDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest


    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions, padding_mask = self.get_emissions(models, encoder_input)
        return self.decode(emissions, padding_mask)

    def get_emissions(self, models, encoder_input):
        """Run encoder and normalize emissions"""
        # encoder_out = models[0].encoder(**encoder_input)
        encoder_out = models[0](**encoder_input)
        print(encoder_out["encoder_out"].size())
        emissions = models[0].get_normalized_probs(encoder_out, log_probs=True)
        return emissions.transpose(0, 1).float().contiguous(), encoder_out["encoder_padding_mask"]

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        return torch.LongTensor(list(idxs))


    def decode(self, emissions, padding_mask):
        B, T, N = emissions.size()
        print(padding_mask)
        print(padding_mask.size())
        mask = 1 - padding_mask.float().unsqueeze(dim=2).repeat(1, 1, N)
        lprobs = torch.sum(emissions.mul(mask), dim=0)
        results = torch.argmax(lprobs, dim=1).unsqueeze(1)
        print("B:", B)
        return [
                [{"tokens": self.get_tokens(results[b].tolist()), "score": 0}]
                for b in range(B)
                ]


