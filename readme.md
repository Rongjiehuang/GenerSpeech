# GenerSpeech: Towards Style Transfer for Generalizable Out-Of-Domain Text-to-Speech

#### Rongjie Huang, Yi Ren, Jinglin Liu, Chenye Cui, Zhou Zhao | Zhejiang University, Sea AI Lab

PyTorch Implementation of [GenerSpeech (NeurIPS'22)](https://arxiv.org/abs/2205.07211): a text-to-speech model towards high-fidelity zero-shot style transfer of OOD custom voice.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2205.07211)
[![GitHub Stars](https://img.shields.io/github/stars/Rongjiehuang/GenerSpeech?style=social)](https://github.com/Rongjiehuang/GenerSpeech)
![visitors](https://visitor-badge.glitch.me/badge?page_id=Rongjiehuang/GenerSpeech)

We provide our implementation and pretrained models in this repository.

Visit our [demo page](https://generspeech.github.io/) for audio samples.

## News
- December, 2022: **[GenerSpeech](https://arxiv.org/abs/2205.07211) (NeurIPS 2022)** released at Github.

## Key Features
- **Multi-level Style Transfer** for expressive text-to-speech.
- **Enhanced model generalization** to out-of-distribution (OOD) style reference.

## Quick Started
We provide an example of how you can generate high-fidelity samples using GenerSpeech.

To try on your own dataset, simply clone this repo in your local machine provided with NVIDIA GPU + CUDA cuDNN and follow the below instructions.

### Support Datasets and Pretrained Models
You can use pretrained models we provide [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/rongjiehuang_zju_edu_cn/EicJK9PNylNEl5fUlFRBExIBzK2MmKjuGSbt8n4HztMv6A?e=h6r8vM). Details of each folder are as in follows:

| Model       | Dataset (16 kHz) | Discription                                                              | 
|-------------|------------------|--------------------------------------------------------------------------|
| GenerSpeech | LibriTTS,ESD     | Acousitic model [(config)](modules/GenerSpeech/config/generspeech.yaml) |
| HIFI-GAN    | LibriTTS,ESD     | Neural Vocoder                                                           |
| Encoder     | /                | Emotion Encoder                                                   |

More supported datasets are coming soon.

### Dependencies

A suitable [conda](https://conda.io/) environment named `generspeech` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate generspeech
```

### Multi-GPU
By default, this implementation uses as many GPUs in parallel as returned by `torch.cuda.device_count()`. 
You can specify which GPUs to use by setting the `CUDA_DEVICES_AVAILABLE` environment variable before running the training module.


## Inference towards style transfer of custom voice
Here we provide a speech synthesis pipeline using GenerSpeech. 

1. Prepare **GenerSpeech** (acoustic model): Download and put checkpoint at `checkpoints/GenerSpeech`
2. Prepare **HIFI-GAN** (neural vocoder): Download and put checkpoint at `checkpoints/trainset_hifigan`
3. Prepare **Emotion Encoder**: Download and put checkpoint at `checkpoints/Emotion_encoder.pt`
4. Prepare **dataset**: Download and put [statistical files](https://zjueducn-my.sharepoint.com/:f:/g/personal/rongjiehuang_zju_edu_cn/EkewjLpkfoJBgOBr5F_59EgBy_twkdZ1yTmtL4HafBJqwg?e=hu671b) at `data/binary/training_set`
5. Prepare **path/to/reference_audio (16k)**: By default, GenerSpeech uses **[ASR](https://huggingface.co/facebook/wav2vec2-base-960h) + [MFA](https://montreal-forced-aligner.readthedocs.io/)** to obtain the text-speech alignment from reference.
```bash
CUDA_VISIBLE_DEVICES=$GPU python inference/GenerSpeech.py --config modules/GenerSpeech/config/generspeech.yaml  --exp_name GenerSpeech --hparams="text='here we go',ref_audio='assets/0011_001570.wav'"
```

Generated wav files are saved in `infer_out` by default.<br>

# Train your own model

### Data Preparation and Configuration ##
1. Set `raw_data_dir`, `processed_data_dir`, `binary_data_dir` in the config file, and download dataset to `raw_data_dir`.
2. Check `preprocess_cls` in the config file. The dataset structure needs to follow the processor `preprocess_cls`, or you could rewrite it according to your dataset. We provide a Libritts processor as an example in `modules/GenerSpeech/config/generspeech.yaml`
3. Download global emotion encoder to `emotion_encoder_path`. For more details, please refer to [this branch](https://github.com/Rongjiehuang/GenerSpeech/tree/encoder).
4. Preprocess Dataset 
```bash
# Preprocess step: unify the file structure.
python data_gen/tts/bin/preprocess.py --config $path/to/config
# Align step: MFA alignment.
python data_gen/tts/bin/train_mfa_align.py --config $path/to/config
# Binarization step: Binarize data for fast IO.
CUDA_VISIBLE_DEVICES=$GPU python data_gen/tts/bin/binarize.py --config $path/to/config
```

You could also build a dataset via [NATSpeech](https://github.com/NATSpeech/NATSpeech), which shares a common MFA data-processing procedure.
We also provide our processed dataset (16kHz LibriTTS+ESD) [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/rongjiehuang_zju_edu_cn/EicJK9PNylNEl5fUlFRBExIBzK2MmKjuGSbt8n4HztMv6A?e=h6r8vM).



### Training GenerSpeech
```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config modules/GenerSpeech/config/generspeech.yaml  --exp_name GenerSpeech --reset
```

### Inference using GenerSpeech

```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config modules/GenerSpeech/config/generspeech.yaml  --exp_name GenerSpeech --infer
```

## Acknowledgements
This implementation uses parts of the code from the following Github repos:
[FastDiff](https://github.com/Rongjiehuang/FastDiff),
[NATSpeech](https://github.com/NATSpeech/NATSpeech),
as described in our code.

## Citations ##
If you find this code useful in your research, please cite our work:
```bib
@inproceedings{huanggenerspeech,
  title={GenerSpeech: Towards Style Transfer for Generalizable Out-Of-Domain Text-to-Speech},
  author={Huang, Rongjie and Ren, Yi and Liu, Jinglin and Cui, Chenye and Zhao, Zhou},
  booktitle={Advances in Neural Information Processing Systems}
}
```

## Disclaimer ##
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

