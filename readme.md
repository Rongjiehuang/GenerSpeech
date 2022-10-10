# GenerSpeech: Towards Style Transfer for Generalizable Out-Of-Domain Text-to-Speech

#### Rongjie Huang, Yi Ren, Jinglin Liu, Chenye Cui, Zhou Zhao

PyTorch Implementation of [GenerSpeech (NeurIPS'22)](https://arxiv.org/abs/2205.07211): a text-to-speech model towards high-fidelity zero-shot style transfer of OOD custom voice.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2205.07211)
[![GitHub Stars](https://img.shields.io/github/stars/Rongjiehuang/GenerSpeech?style=social)](https://github.com/Rongjiehuang/GenerSpeech)
![visitors](https://visitor-badge.glitch.me/badge?page_id=Rongjiehuang/GenerSpeech)

**We provide the implementation of the wav2vec 2.0 global encoder as open source in this branch.**

# Train your own model

## Requirements

This repo also includes the source code of **fairseq**.

Run `pip install --editable ./` to install the necessary packages.

A GPU is mandatory, but you don't necessarily need a high tier GPU if you only want to use the toolbox.


### Data Preparation and Configuration ##
1. Set `DATA_DIR`, `W2V_PATH`, `SAVE_DIR` in the bash file
2. Download dataset (IEMOCAP/VoxCeleb1) to `/path/to/waves`.
3. Create manifest `.tsv`
```
python examples/wav2vec/wav2vec_manifest.py /path/to/waves --dest $DATA_DIR --ext $ext --valid-percent $valid
```
$ext should be set to flac, wav, or whatever format your dataset happens to use that soundfile can read.
$valid should be set to some reasonable percentage (like 0.01) of training data to use for validation.

4. Create label `.ltr`. You could refer to the [example](exp/data/VOX1).
5. Download pre-trained wav2vec 2.0 base model [here](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md) to `W2V_PATH`.



### Training 
```bash
bash run.sh   # Train speaker encoder
bash run_emotion.sh    # Train emotion encoder
```

### Evaluation

```bash
bash evaluate.sh   # Evaluate speaker encoder
bash evaluate_emotion.sh    # Evaluate emotion encoder
```

### Inference

```bash
python generate_embedding $data_path  \
--task audio_pretraining_sid --path $model_path --criterion  classification_amsoftmax
```

## Acknowledgements
We appreciate Zhiyun Fan for sharing the source code of ```Exploring wav2vec 2.0 on speaker verification and language identification```, which is the codebase of this repo.

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
