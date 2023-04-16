#/usr/bin/bash

export MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.1 --time_embed_dim 128"
export DIFFUSION_FLAGS="--diffusion_steps 50 --noise_schedule cosine"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 100 --save_interval 5000"
export CHECKPOINT_FLAGS="--resume_checkpoint log/model020000.pt"
export OPENAI_LOGDIR="log"
export OPENAI_LOG_FORMAT="stdout"

python scripts/image_train.py --data_dir cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $CHECKPOINT_FLAGS
