#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018  Tsinghua University (Author: Zhiyuan Tang)


import sys, collections

if len(sys.argv) != 3:

  print('usage: prepare_trials.py <enroll-dir> <test-dir>')

  sys.exit()

enroll_dir = sys.argv[1]
test_dir = sys.argv[2]

lang2num_file = "/mnt/lustre/xushuang2/zyfan/program/code/wav2vec2.0/fairseq/exp/data/AP18/dict.ltr.txt"

lang2num = {}
with open(lang2num_file, 'r') as fo:
    for line in fo:
        lang = line.strip().split()
        lang2num[lang[0]]=lang[1]


# lang list in enroll dir
lang_list = []
with open(enroll_dir + '/utt2lang', 'r') as langs:
  for line in langs:
    lang_list.append(line.strip().split()[1])
lang_list = list(set(lang_list))
lang_list.sort()


# utt2lang dict in test dir
lang_dict = collections.OrderedDict()
with open(test_dir + '/utt2lang', 'r') as lang_ids:
  for lang_id in [line.strip().split() for line in lang_ids]:
     lang_dict[lang_id[0]] = lang_id[1]


# generate trials
trial = open(test_dir + '/trials', 'w')
for i in lang_dict.keys():
  for j in lang_list:
    if lang2num[j] == lang_dict[i]:
      trial.write(lang2num[j] + ' ' + i + ' ' + 'target' + '\n')
    else:
      trial.write(lang2num[j] + ' ' + i + ' ' + 'nontarget' + '\n')


print('Finished preparing trials.')
