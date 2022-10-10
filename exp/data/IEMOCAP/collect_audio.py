# Try for one file first
import librosa
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
from tqdm import tqdm
import pickle

import IPython.display
import librosa.display
ms.use('seaborn-muted')

import pandas as pd
import math
from scipy.io import wavfile
labels_df = pd.read_csv('exp/data/IEMOCAP/emotion_totallabel.csv')
iemocap_dir = '/Old/huangrongjie/dataset/dataset/IEMOCAP_full_release/'
save_path = 'exp/data/IEMOCAP/processed_data/'

sr = 44100
audio_vectors = {}
for sess in range(1, 6):  # using one session due to memory constraint, can replace [5] with range(1, 6)
    wav_file_path = '{}Session{}/dialog/wav/'.format(iemocap_dir, sess)
    orig_wav_files = os.listdir(wav_file_path)
    for orig_wav_file in tqdm(orig_wav_files):
        try:
            orig_wav_vector, _sr = librosa.load(wav_file_path + orig_wav_file, sr=sr)
            orig_wav_file, file_format = orig_wav_file.split('.')
            for index, row in labels_df[labels_df['wav_file'].str.contains(orig_wav_file)].iterrows():
                start_time, end_time, truncated_wav_file_name, emotion, val, act, dom = row['start_time'], row['end_time'], row['wav_file'], row['emotion'], row['val'], row['act'], row['dom']
                start_frame = math.floor(start_time * sr)
                end_frame = math.floor(end_time * sr)
                truncated_wav_vector = orig_wav_vector[start_frame:end_frame + 1]

                save_audio = truncated_wav_vector * 32767
                wavfile.write(save_path + truncated_wav_file_name + ".wav", sr, save_audio.astype(np.int16))
        except:
            print('An exception occured for {}'.format(orig_wav_file))