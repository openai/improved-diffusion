#/usr/bin/bash

export MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.1 --time_embed_dim 128 --early_stop 15 --sigma_small False"
export DIFFUSION_FLAGS="--diffusion_steps 50 --noise_schedule cosine"
export SAMPLE_FLAGS="--num_samples 50000 --batch_size 64 --model_path log/ema_0.9999_150000.pt --residual_path log/residual_ema_0.9999_150000.pt"
export OPENAI_LOGDIR="sample_log_150k"
export OPENAI_LOG_FORMAT="stdout"

python scripts/image_sample.py $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS