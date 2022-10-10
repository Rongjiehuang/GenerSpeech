model_path=exp/wav2vev2_small_finetune_IEMOCAP_amsoftmax/checkpoint_last.pt
data_path=/home1/huangrongjie/Project/Emotion_encoder/ESD_emo_small.tsv

CUDA_VISIBLE_DEVICES=0  python test_emotion.py $data_path  \
--task audio_pretraining_sid --path $model_path --criterion  classification_amsoftmax