#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run inference for pre-processed data with a trained model.
"""

import logging
import math
import numpy, math, pdb, sys, random
from tqdm import tqdm
import time, os, itertools, shutil, importlib
from scipy.io import wavfile
import os
import sys
from glob import glob
from sklearn import metrics
import soundfile as sf
from pathlib import Path
#import sentencepiece as spm
import torch
from fairseq import checkpoint_utils, options, utils, tasks
from fairseq.logging import meters, progress_bar
from fairseq.utils import import_user_module

import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_model(filename, arg_overrides=None, task=None):
    if not os.path.exists(filename):
        raise IOError("Model file not found: {}".format(filename))
    state = checkpoint_utils.load_checkpoint_to_cpu(filename, arg_overrides)
    #state["model"].pop('w2v_encoder.W')
    #state["model"].pop('w2v_encoder.W.bias')
    args = state["args"]
    if task is None:
        task = tasks.setup_task(args)
        # build model for ensemble
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=True)
    return model


def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio,sample_rate = sf.read(filename)

    feats_v0 = torch.from_numpy(audio).float()
    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = math.floor((max_audio - audiosize + 1) / 2)
        audio = numpy.pad(audio, (shortage, shortage), 'constant', constant_values=0)
        audiosize = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random() * (audiosize - max_audio))])
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])
    feat = numpy.stack(feats, axis=0)
    feat = torch.FloatTensor(feat)
    return feat;

def evaluateFromList(model):

    model.eval();
    spk_embed = {}
    files = list(Path('/home1/huangrongjie/Project/NeuralSeq/data/processed/LibriTTS_16k/mfa_inputs').glob('*/*.wav'))

    ## Save all features to file
    for idx, file in tqdm(enumerate(files)):
        #print(111,idx,file,os.path.join(test_path,file))
        inp1 = loadWAV(str(file), 300, evalmode=True, num_eval=10)
        inp1=inp1.cuda()
        #ref_feat = self.__S__.forward(inp1).detach().cpu()
        padding = torch.BoolTensor(inp1.shape).fill_(False).cuda()
        encoder_inp1 = {'source':inp1,'padding_mask':padding}

        net_output = model(**encoder_inp1)
        # ref_feat = torch.mean(net_output["features_out"].detach().cpu(), dim=1)
        ref_feat = torch.mean(net_output["encoder_out"], dim=0).detach().cpu().numpy()
        item_name = file.stem[8:]
        spk_embed[item_name] = ref_feat
    numpy.save('/home1/huangrongjie/Project/NeuralSeq/data/processed/LibriTTS_16k/spk_embed.npy', spk_embed)
    return

def main(args):
    import_user_module(args)
    logger.info(args)
    use_cuda = torch.cuda.is_available()
    task = tasks.setup_task(args)
    model = load_model(args.path,arg_overrides=eval(args.model_overrides),  # noqa
        task=task,)
    if use_cuda:
        model.cuda()
    evaluateFromList(model)
    print('End!')


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)

if __name__ == "__main__":
    cli_main()
