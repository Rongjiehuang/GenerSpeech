model_path=exp/wav2vev2_small_finetune_VOX1_amsoftmax_2/checkpoint_last.pt
data_path=exp/data/VOX1/test_list.txt

CUDA_VISIBLE_DEVICES=1  python test_speaker.py $data_path  \
--task audio_pretraining_sid --path $model_path --criterion  classification_amsoftmax