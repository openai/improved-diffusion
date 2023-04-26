#/usr/bin/bash

export MODEL_FLAGS="--use_residual True --return_residual_value True --image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.1 --time_embed_dim 128 --early_stop 10 --sigma_small True"
export DIFFUSION_FLAGS="--diffusion_steps 50 --noise_schedule exponential"
export SAMPLE_FLAGS="--num_samples 50000 --batch_size 128 --model_path log_exp_T50/ema_0.9999_050000.pt --residual_path log_exp_T50/residual_ema_0.9999_050000.pt"
export OPENAI_LOGDIR="exp_sample_log_50k"
export OPENAI_LOG_FORMAT="stdout"

python scripts/image_sample.py $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS