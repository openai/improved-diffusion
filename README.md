# improved-diffusion

This is the codebase for [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672).

# Usage

This section of the README walks through how to train and sample from a model.

## Installation

Clone this repository and navigate to it in your terminal. Then run:

```
pip install -e .
```

This should install the `improved_diffusion` python package that the scripts depend on.

## Preparing Data

The training code reads images from a directory of image files. In the [datasets](datasets) folder, we have provided instructions/scripts for preparing these directories for ImageNet, LSUN bedrooms, and CIFAR-10.

For creating your own dataset, simply dump all of your images into a directory with ".jpg", ".jpeg", or ".png" extensions. If you wish to train a class-conditional model, name the files like "mylabel1_XXX.jpg", "mylabel2_YYY.jpg", etc., so that the data loader knows that "mylabel1" and "mylabel2" are the labels. Subdirectories will automatically be enumerated as well, so the images can be organized into a recursive structure (although the directory names will be ignored, and the underscore prefixes are used as names).

The images will automatically be scaled and center-cropped by the data-loading pipeline. Simply pass `--data_dir path/to/images` to the training script, and it will take care of the rest.

## Training

To train your model, you should first decide some hyperparameters. We will split up our hyperparameters into three groups: model architecture, diffusion process, and training flags. Here are some reasonable defaults for a baseline:

```
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
```

Here are some changes we experiment with, and how to set them in the flags:

 * **Learned sigmas:** add `--learn_sigma True` to `MODEL_FLAGS`
 * **Cosine schedule:** change `--noise_schedule linear` to `--noise_schedule cosine`
 * **Reweighted VLB:** add `--use_kl True` to `DIFFUSION_FLAGS` and add `--schedule_sampler loss-second-moment` to  `TRAIN_FLAGS`.
 * **Class-conditional:** add `--class_cond True` to `MODEL_FLAGS`.

Once you have setup your hyper-parameters, you can run an experiment like so:

```
python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

You may also want to train in a distributed manner. In this case, run the same command with `mpiexec`:

```
mpiexec -n $NUM_GPUS python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

When training in a distributed manner, you must manually divide the `--batch_size` argument by the number of ranks. In lieu of distributed training, you may use `--microbatch 16` (or `--microbatch 1` in extreme memory-limited cases) to reduce memory usage.

The logs and saved models will be written to a logging directory determined by the `OPENAI_LOGDIR` environment variable. If it is not set, then a temporary directory will be created in `/tmp`.

## Sampling

The above training script saves checkpoints to `.pt` files in the logging directory. These checkpoints will have names like `ema_0.9999_200000.pt` and `model200000.pt`. You will likely want to sample from the EMA models, since those produce much better samples.

Once you have a path to your model, you can generate a large batch of samples like so:

```
python scripts/image_sample.py --model_path /path/to/model.pt $MODEL_FLAGS $DIFFUSION_FLAGS
```

Again, this will save results to a logging directory. Samples are saved as a large `npz` file, where `arr_0` in the file is a large batch of samples.

Just like for training, you can run `image_sample.py` through MPI to use multiple GPUs and machines.

You can change the number of sampling steps using the `--timestep_respacing` argument. For example, `--timestep_respacing 250` uses 250 steps to sample. Passing `--timestep_respacing ddim250` is similar, but uses the uniform stride from the [DDIM paper](https://arxiv.org/abs/2010.02502) rather than our stride.

To sample using [DDIM](https://arxiv.org/abs/2010.02502), pass `--use_ddim True`.
