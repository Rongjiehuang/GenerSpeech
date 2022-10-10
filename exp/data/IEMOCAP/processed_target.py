
import json
from pathlib import Path
from glob import glob
from tqdm import tqdm
import os
import random
import soundfile
import pandas as pd

root = "exp/data/IEMOCAP/processed_data/"
savepath = "exp/data/IEMOCAP"
rand = random.Random(42)
labels_df = pd.read_csv('exp/data/IEMOCAP/emotion_totallabel.csv')

# manifest
with open(os.path.join(savepath, 'train.tsv'), 'w') as train_f, \
        open(os.path.join(savepath, 'valid.tsv'), 'w') as valid_f,\
        open(os.path.join(savepath, 'train.ltr'), 'w') as train_emo_l, \
        open(os.path.join(savepath, 'valid.ltr'), 'w') as valid_emo_l:
    print(root, file=train_f)
    print(root, file=valid_f)

    total_length = len(labels_df['wav_file'])
    for i in tqdm(range(total_length)):
        emo = labels_df['emotion'][i]

        file_path = labels_df['wav_file'][i] + ".wav"
        if not os.path.exists(root + file_path) or emo == 'xxx':
            continue



        frames = soundfile.info(root + file_path).frames
        if rand.random() > 0.01:
            dest = train_f
            dest_emo = train_emo_l
        else:
            dest = valid_f
            dest_emo = valid_emo_l

        print('{}\t{}'.format(file_path, frames), file=dest)
        print('{}'.format(emo), file=dest_emo)
