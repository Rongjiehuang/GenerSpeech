from encoder.params_model import model_embedding_size as speaker_embedding_size
from encoder import inference as encoder
from pathlib import Path
import numpy as np
import librosa
import argparse
import torch
import sys
import os
import glob


if __name__ == '__main__':
    # Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_dir", type=Path,
                        default="synthesizer/saved_models/logs-pretrained/",
                        help="Directory containing the synthesizer model")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--low_mem", action="store_true", help="If True, the memory used by the synthesizer will be freed after each use. Adds large "
                        "overhead but allows to save some GPU memory for lower-end GPUs.")
    parser.add_argument("--no_sound", action="store_true",
                        help="If True, audio won't be played.")
    parser.add_argument("-a", "--audio_fpath", type=Path,
                        default="audios/",
                        help="Path to wave files")
    parser.add_argument("-m", "--embed_fpath", type=Path,
                        default="audios/embeds/",
                        help="Path to save embeddings")

    args = parser.parse_args()
    if not args.no_sound:
        import sounddevice as sd
    
    # File path where wav files are stored
    #path_wav = '/home/dipjyoti/speaker_embeddings_GE2E/audios/'

    # File path where generated speaker embeddings are stored
    #path_embed = '/home/dipjyoti/speaker_embeddings_GE2E/audios/embeds/'



    # Load the models one by one.
    print("Preparing the encoder...")
    encoder.load_model(args.enc_model_fpath)
    print("Insert the wav file name...")
    try:
        # Get the reference audio filepath

        for filename in glob.glob(os.path.join(args.audio_fpath, '*.wav')):
            print(filename)
        # Computing the embedding
        # First, we load the wav using the function that the speaker encoder provides. This is
        # important: there is preprocessing that must be applied.

        # The following two methods are equivalent:
        # - Directly load from the filepath:
            preprocessed_wav = encoder.preprocess_wav(filename)

        # Then we derive the embedding. There are many functions and parameters that the
        # speaker encoder interfaces. These are mostly for in-depth research. You will typically
        # only use this function (with its default parameters):
            embed = encoder.embed_utterance(
                preprocessed_wav)
            embed_path = args.embed_fpath / \
                filename.split('/')[-1].replace('.wav', '.npy')
            np.save(embed_path, embed)
            print("Created the embeddings")

    except Exception as e:
        print("Caught exception: %s" % repr(e))
        print("Restarting\n")
