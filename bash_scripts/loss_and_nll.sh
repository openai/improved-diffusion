#/usr/bin/bash

export MODEL_FLAGS="--use_residual False --image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.1 --time_embed_dim 128 --sigma_small False"
export DIFFUSION_FLAGS="--diffusion_steps 50 --noise_schedule linear"
export NLL_FLAGS="--data_dir cifar_train --num_samples 50000 --batch_size 64 --model_path log_base/ema_0.9999_200000.pt --residual_path log_base/residual_ema_0.9999_200000.pt"
export OPENAI_LOGDIR="nll_log_base_large"
export OPENAI_LOG_FORMAT="stdout"

python scripts/image_nll.py $MODEL_FLAGS $DIFFUSION_FLAGS $NLL_FLAGS