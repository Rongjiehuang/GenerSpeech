SAVE_DIR=/home1/huangrongjie/Project/fairseq-sid-simplify-to-bak/exp/wav2vev2_small_finetune_VOX1_amsoftmax_2
W2V_PATH=/home1/huangrongjie/Project/fairseq-sid-simplify-to-bak/exp/wav2vec.pt
DATA_DIR=/home1/huangrongjie/Project/fairseq-sid-simplify-to-bak/exp/data/VOX1

if [ ! -d $SAVE_DIR ]; then 
    mkdir $SAVE_DIR -p
fi

cp $0 $SAVE_DIR

CUDA_VISIBLE_DEVICES=0  python -u train.py $DATA_DIR --save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR --fp16 --train-subset train --valid-subset valid --no-epoch-checkpoints --best-checkpoint-metric loss --num-workers 4 \
--max-update 60000 --sentence-avg --task audio_pretraining_sid --arch wav2vec_amsoftmax --w2v-path $W2V_PATH \
--labels ltr --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5  \
--feature-grad-mult 0.0 --freeze-finetune-updates 0 --validate-after-updates 20000  --validate-interval 50  --optimizer adam \
--adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 2e-05 --lr-scheduler tri_stage --warmup-steps 8000 --hold-steps 2000 \
--decay-steps 20000 --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion  classification_amsoftmax \
--attention-dropout 0.0  --max-tokens 1000000 --seed 2337 --ddp-backend no_c10d --update-freq 4  \
--log-interval 10 --log-format simple --save-interval-updates 200 --validate-interval-updates 800 --max-sample-size 160000
