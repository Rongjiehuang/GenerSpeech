# GenerSpeech: Towards Style Transfer for Generalizable Out-Of-Domain Text-to-Speech

#### Rongjie Huang, Yi Ren, Jinglin Liu, Chenye Cui, Zhou Zhao

PyTorch Implementation of [GenerSpeech (NeurIPS'22)](https://arxiv.org/abs/2205.07211): a text-to-speech model towards high-fidelity zero-shot style transfer of OOD custom voice.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2205.07211)
[![GitHub Stars](https://img.shields.io/github/stars/Rongjiehuang/GenerSpeech?style=social)](https://github.com/Rongjiehuang/GenerSpeech)
![visitors](https://visitor-badge.glitch.me/badge?page_id=Rongjiehuang/GenerSpeech)

We observe that the CNN-based speech encoder works well. To simplify overall pipeline, we provide our Global Emotion/Speaker Encoder re-implementation and pretrained models as open source in this branch.

For implementation of the wav2vec 2.0 global encoder, please refer to the wav2vec branch [here](https://github.com/Rongjiehuang/GenerSpeech/tree/wav2vec).

### Support Datasets and Pretrained Models

Download the pretrained model we provide [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/rongjiehuang_zju_edu_cn/Eko36YpkWdhKjLxAptY-idgBN0z5OC8q7eO0n3wYZEq2hA?e=gROUjE). 

| Model       | Discription              | 
|-------------|--------------------------|
| Encoder     | Global Emotion Encoder   |

For other datasets, please fine-tune the pretrained models for better results.

## Inference
### Emotion Embedding
```python
from encoder import inference as EmotionEncoder
from encoder.inference import embed_utterance as Embed_utterance
from encoder.inference import preprocess_wav

emo_encoder.load_model(path/to/emotion_encoder)
processed_wav = preprocess_wav(path/to/wave)
emo_embed = Embed_utterance(processed_wav)
```

### Speaker Embedding
```python
from resemblyzer import VoiceEncoder, preprocess_wav

spk_encoder = VoiceEncoder()
processed_wav = preprocess_wav(path/to/wave)
spk_embed = encoder.embed_utterance(processed_wav)
```


## Train your own model

### Requirements

**Python 3.6 +**.

Run `pip install -r requirements.txt` to install the necessary packages.

A GPU is mandatory, but you don't necessarily need a high tier GPU if you only want to use the toolbox.

### Datasets

Ideally, all your datasets are kept under a same directory i.e., ```<datasets_root>```. All prepreprocessing scripts will, by default, output the clean data to a new directory SV2TTS created in your datasets root directory. Inside this directory will be created a directory for the encoder.


### Preprocessing and training
```
python encoder_preprocess.py <datasets_root>
python encoder_train.py my_run <datasets_root>/SV2TTS/encoder
```

### Generate speaker embeddings

```
python generate_embeddings.py
```

## Acknowledgements
This implementation uses parts of the code from the following Github repos:
[Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning),
[NATSpeech](https://github.com/NATSpeech/NATSpeech),
as described in our code.

## Citations ##
If you find this code useful in your research, please cite our work:
```bib
@article{huang2022generspeech,
  title={GenerSpeech: Towards Style Transfer for Generalizable Out-Of-Domain Text-to-Speech Synthesis},
  author={Huang, Rongjie and Ren, Yi and Liu, Jinglin and Cui, Chenye and Zhao, Zhou},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Disclaimer ##
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.
