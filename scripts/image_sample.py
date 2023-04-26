"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    create_residual_connection_net,
    model_and_diffusion_defaults,
    residual_connection_net_defaults,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    if args.use_residual:
        residual_connection_net = create_residual_connection_net(
            **args_to_dict(args, residual_connection_net_defaults())
        )
        residual_connection_net.load_state_dict(
            dist_util.load_state_dict(args.residual_path, map_location="cpu")
        )
        residual_connection_net.to(dist_util.dev())
        residual_connection_net.eval()
    else:
        residual_connection_net = None

    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    
    #################################
    ## Create residual value array ##
    #################################
    if args.return_residual_value:
        all_residual_values = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        sample_fn = diffusion.p_sample_loop_early_stop
        sample_output = sample_fn(
            model,
            residual_connection_net,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            end_step=args.early_stop,
            return_residual_value=args.return_residual_value,
            )

        #################
        ## Save images ##
        #################
        sample = sample_output["pred_xstart"]
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 1, 3, 4, 2)
        sample = sample.contiguous()
        
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

        ##########################
        ## Save residual values ##
        ##########################
        if args.return_residual_value:
            residual_value = sample_output["residual_value"]
            gathered_values = [th.zeros_like(residual_value) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_values, residual_value)
            all_residual_values.extend([values.cpu().numpy() for values in gathered_values])

    arr = np.concatenate(all_images, axis=1)
    arr = arr[:, : args.num_samples]

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    if args.return_residual_value:
        arr = np.concatenate(all_residual_values, axis=1)
        arr = arr[:, : args.num_samples]
        
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(), f"residuals_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        residual_path="",
        early_stop=1,
        return_residual_value=False,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(residual_connection_net_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()