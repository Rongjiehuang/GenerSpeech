#/usr/bin/python

#author: fanzhiyun
#date: 20200216

import sys

score_file = sys.argv[1]
label_file = sys.argv[2]
trial_file = sys.argv[3]

trials = open(trial_file, 'r').readlines()


lang_dict = {}
with open(label_file, 'r') as lang_ids:
  for lang_id in [line.strip().split() for line in lang_ids]:
     lang_dict[lang_id[0]] = lang_id[1]


spkrutt2target = {}
with open(trial_file, 'r') as foo:
  for line in foo:
    spkr, utt, target = line.strip().split()
    spkrutt2target[spkr+utt]=target



with open(score_file, 'r') as fo:
  lines = fo.readlines()
  langs = lines[0].strip().split()
  for line in lines[1:]:
    utt = line.strip().split()[0]
    scores = line.strip().split()[1:]
    for i in range(len(scores)):
      print(scores[i], spkrutt2target[langs[i]+utt])
