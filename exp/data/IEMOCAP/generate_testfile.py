
import json
from pathlib import Path
from glob import glob
from tqdm import tqdm
import os
import random
from random import choice

meta_paths = list(Path("/home1/huangrongjie/Project/s3prl/s3prl/downstream/emotion/meta_data/").glob("*/*.json"))
meta_data = []
for meta_path in tqdm(meta_paths):
    with open(meta_path, 'r') as f:
        data = json.load(f)
    meta_data += data['meta_data']

root = "/Old/huangrongjie/dataset/dataset/IEMOCAP_wav/"
savepath = "exp/data/IEMOCAP/test"
rand = random.Random(42)

emos = {}
spks = {}

for data in meta_data:
    file_path = data['path']
    emo = data['label']
    spk = data['speaker']

    if emo not in emos:
        emos[emo] = [file_path]
    else:
        emos[emo].append(file_path)

    if spk not in spks:
        spks[spk] = [file_path]
    else:
        spks[spk].append(file_path)

with open(os.path.join(savepath, 'IEMOCAP_emo.tsv'), 'w') as f:
    emo_name = list(emos.keys())
    emo_list = [i for i in range(len(emo_name))]
    for i in tqdm(emo_list):
        files = emos[emo_name[i]]
        for file in files:
            same_emo_file = file
            while same_emo_file == file:
                same_emo_file = choice(files)
            different_emo = i
            while different_emo == i:
                different_emo = choice(emo_list)
            different_emo_file = choice(emos[emo_name[different_emo]])

            print('1\t{}\t{}'.format(file, same_emo_file), file=f)
            print('0\t{}\t{}'.format(file, different_emo_file), file=f)
