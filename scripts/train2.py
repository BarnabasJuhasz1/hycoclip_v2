#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

# Modified from github.com/facebookresearch/meru

"""
Train a HyCoCLIP, MERU or CLIP model based on parameters specified by a config file.
"""
import os

from pydantic import InstanceOf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# uncomment the following line to disable torch.compile()
# os.environ["TORCH_COMPILE_DISABLE"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import random
from pathlib import Path
import socket
import wandb
import ast

import torch
import numpy as np
from loguru import logger
from omegaconf import OmegaConf
# from torch.cuda import amp
import torch.amp as amp
from torch.utils.tensorboard import SummaryWriter

import hycoclip.utils.distributed as dist
from hycoclip.config import LazyConfig, LazyFactory
from hycoclip.tokenizer import Tokenizer
from hycoclip.utils.checkpointing import CheckpointManager
from hycoclip.utils.timer import Timer
from hycoclip.models import HyCoCLIP

# from hycoclip.models import HyCoCLIP, HyCoCLIP_Re_Weight, HyCoCLIP_Re_Modulate, HyCoCLIP_Re_Combined
# from hycoclip.new_models.re_weight_DinContrastive import HyCoCLIP_Re_Weight_DinContrastive
# from hycoclip.new_models.re_weight_withoutD import HyCoCLIP_Re_Weight_withoutD

# from finetuning.wandb import initialize_wandb_logger 

#Disable Huggingface’s online requests
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


# fmt: off
parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("--wandb", action="store_true", help="Whether to log to Weights & Biases.")
parser.add_argument("--wandb-config", type=str, help="WandB configs [str but in dict shape]")
parser.add_argument("--grad-clip", action="store_true", help="Whether to perform gradient clipping.")
parser.add_argument("--config", help="Path to a .py config file.")
parser.add_argument(
    "--output-dir", default="./output",
    help="Path to a directory to save checkpoints and job logs.",
)
parser.add_argument(
    "--resume", action="store_true",
    help="Whether to resume training from `--output-dir`. This script will find "
    "the last saved checkpoint and resume training. It is user's responsibility "
    "to provide matching config file in `--config`.",
)
parser.add_argument(
    "--checkpoint-period", type=int, default=5000, help="Checkpoint saving period."
)
parser.add_argument(
    "--log-period", type=int, default=100,
    help="Log to stdout/tensorboard periodically (only main process).",
)
parser.add_argument(
    "--num-machines", type=int, default=1,
    help="Number of machines used in distributed training.",
)
parser.add_argument(
    "--num-gpus", type=int, default=0, help="Number of GPUs per machine."
)
parser.add_argument(
    "--machine-rank", type=int, default=0,
    help="Integer in [0, num_machines) to specifying machine ID.",
)
_random_port = random.randint(2000, 19999)
parser.add_argument(
    "--dist-url", default=f"tcp://127.0.0.1:{_random_port}",
    help="URL of the main process in distributed training, it defaults to "
    "localhost for single-machine training.",
)
parser.add_argument(
    "overrides", nargs="...", default=[], help="Config overrides (key-value pairs)."
)
# fmt: on


def main(_A: argparse.Namespace):
    # -------------------------------------------------------------------------
    #   BASIC SETUP FOR TRAINING JOB.
    # -------------------------------------------------------------------------
    # Create a config object and perform common setup.
    _C = LazyConfig.load(_A.config)
    _C = LazyConfig.apply_overrides(_C, _A.overrides)
    
    # Get process rank and world size (assuming distributed is initialized).
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()

    if getattr(_C.train, "seed", None) is None:
        _C.train.seed = int(time.time())

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(_C.train.seed + RANK)
    np.random.seed(_C.train.seed + RANK)
    torch.manual_seed(_C.train.seed + RANK)
    torch.backends.cudnn.deterministic = _C.train.cudnn_deterministic
    torch.backends.cudnn.benchmark = _C.train.cudnn_benchmark
    torch.set_float32_matmul_precision('high')

    # Create output directory and save config in it.
    output_dir = Path(_A.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LazyConfig.save(_C, output_dir / "config.yaml")

    # Create a logger for each process which writes to a separate log-file.
    logger.add(output_dir / f"log-rank{RANK}.txt", format="{time} {level} {message}")

    # Print process info, config and args.
    logger.info(f"Rank of current process: {RANK}. World size: {WORLD_SIZE}")
    logger.info(f"RANK {RANK} using random seed: {_C.train.seed + RANK}")
    logger.info(OmegaConf.to_yaml(_C))

    if dist.is_main_process():
        logger.info(OmegaConf.to_yaml(_C))

        logger.info("Command line args:")
        for arg in vars(_A):
            logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    # -------------------------------------------------------------------------
    #   INSTANTIATE ALL OBJECTS FOR TRAINING.
    # -------------------------------------------------------------------------
    device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if _A.num_gpus != 0
        else torch.device("cpu")
    )
    dataloader = LazyFactory.build_dataloader(_C)
    tokenizer = Tokenizer()

    model = LazyFactory.build_model(_C, device)
    use_boxes = _C.model.use_boxes
    use_hierarchies = _C.model.use_hierarchies

    optimizer = LazyFactory.build_optimizer(_C, model)
    scheduler = LazyFactory.build_lr_scheduler(_C, optimizer)
    scaler = amp.GradScaler(enabled=_C.train.amp)

    checkpoint_manager = CheckpointManager(
        _A.output_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    start_iteration = checkpoint_manager.resume(model_only=False) if _A.resume else 0

    # Create an iterator from dataloader to sample batches perpetually.
    dataloader_iter = iter(dataloader)
    timer = Timer(start_iteration + 1, total_iterations=_C.train.num_iterations)

    # if _A.fast_min_dist_estimation:
    #     from hycoclip import lorentz as L
        
    #     if dist.is_main_process():
    #         all_dists = []
    #         logger.info("Estimating the minimum distance")
    #     with torch.inference_mode():
    #         for i in range(100):
    #             batch = next(dataloader_iter)
    #             tokens = tokenizer(batch["text"])
    #             text_embeddings = model.encode_text(tokens, project=True)
    #             image_embeddings = model.encode_image(batch["image"].to(device), project=True)

    #             pos_dist = L.elementwise_dist(text_embeddings, image_embeddings, model.curv.exp())
                
    #             if dist.is_main_process():
    #                 gathered_dists = dist.gather_across_processes(pos_dist)
    #                 all_dists.extend(gathered_dists)
                    
    #             torch.cuda.synchronize()

    #         if dist.is_main_process():
    #             all_dists = torch.cat(all_dists, dim=0)
    #             median_dist = all_dists.median()
    #             logger.info(f"Estimated median distance: {median_dist.item():.9f}")
        
    #     return

    # Create tensorboard writer, only in main process.
    if dist.is_main_process():
        tboard = SummaryWriter(log_dir=_A.output_dir)
        if _A.wandb:
            wandb_config = ast.literal_eval(_A.wandb_config)
            # Convert OmegaConf config to a regular dict for wandb logging
            wandb_config_dict = OmegaConf.to_container(_C, resolve=True, enum_to_str=True)
            if _A.resume:
                wandb_config['name'] = wandb_config['name'] + '_resumed'
            run = wandb.init(config=wandb_config_dict, **wandb_config)


    # print(f"Using BOXES ({use_boxes}) and HyperHI5 ({use_hyperhi5}) WITH {type(model.module)}!!!!!!!!!!!!!!!!!!!!")

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    for iteration in range(start_iteration + 1, _C.train.num_iterations + 1):
        data_time = time.perf_counter()
        batch = next(dataloader_iter)
        data_time = time.perf_counter() - data_time

        timer.tic()
        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(False):
            with amp.autocast(enabled=_C.train.amp, device_type=device.type):
                # Get image and text (tokens) from batch and pass through model.

                tokens = tokenizer(batch["text"])
                if use_boxes:
                    box_tokens = tokenizer(batch["box_text"])
                    if use_hierarchies:
                        text_hierarchy_tokens = [tokenizer(h) for h in batch["text_hierarchy"]]
                        output_dict = model(batch["image"].to(device),
                                            batch["box_image"].to(device),
                                            tokens,
                                            box_tokens,
                                            text_hierarchy_tokens,
                                            batch["scores"].to(device))
                    
                    else:
                        output_dict = model(batch["image"].to(device),
                                            batch["box_image"].to(device),
                                            tokens,
                                            box_tokens)
                else:
                    output_dict = model(batch["image"].to(device), tokens)
                    

                loss = output_dict["loss"]

            scaler.scale(loss).backward()

            # gradient clipping
            # if _A.grad_clip:
            #     scaler.unscale_(optimizer)
            #     torch.nn.utils.clip_grad_norm_(
            #         model.parameters(), 1.0
            #     )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            timer.toc()

        # Log statistics to terminal and tensorboard.
        if dist.is_main_process():
            if iteration % _A.log_period == 0:
                timer_stats = (
                    f"Iter {timer.iteration} | Time (sec): {data_time:.3f} data, "
                    f"{timer.deltas[-1]:.3f} model | ETA: {timer.eta_hhmm}"
                )

                # Log gradient norms
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2).item()
                        total_norm += param_norm ** 2
                total_norm = total_norm ** 0.5
                
                # if np.isnan(total_norm) or np.isinf(total_norm) or total_norm < 1e-8:
                #     return

                log_str = f"{timer_stats} [GPU {dist.gpu_mem_usage()} MB]"
                for key, value in output_dict["logging"].items():
                    log_str += f" [{key} {value:.3f}]"

                log_str += f" [Grad Norm {total_norm:.3f}] [AMP Scale {scaler.get_scale():.3f}]"
                logger.info(log_str)

                tboard.add_scalar("lr", scheduler.get_last_lr()[0], iteration)
                tboard.add_scalar("amp_scale", scaler.get_scale(), iteration)
                tboard.add_scalar("grad_norm", float(total_norm), iteration)
                for name, _loss in output_dict["logging"].items():
                    tboard.add_scalar(f"train/{name}", float(_loss.mean().item()), iteration)

                if _A.wandb:
                    wandb_log_dict = {
                        "iteration": iteration,
                        "lr": scheduler.get_last_lr()[0],
                        "amp_scale": scaler.get_scale(),
                        "grad_norm": total_norm,
                        **{loss_key: loss_value.mean() for loss_key, loss_value in output_dict["logging"].items() if 'loss' in loss_key},
                    }
                    run.log(wandb_log_dict, step=iteration)

        # Save checkpoint to disk.
        if iteration % _A.checkpoint_period == 0 and dist.is_main_process():
            checkpoint_manager.step(iteration)

    # Save the final checkpoint.
    if dist.is_main_process():
        checkpoint_manager.final_step()
        if _A.wandb:
            run.finish()

if __name__ == "__main__":
    _A = parser.parse_args()
    if _A.num_gpus == 0:
        main(_A)
    else:
        # This will launch `main` and set appropriate CUDA device (GPU ID) as
        # per process (accessed in the beginning of `main`).
        # cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        # stdout = subprocess.check_output(cmd.split())
        # host_name = stdout.decode().splitlines()[0]
        # logger.info(f"Host name: {host_name}")
        # dist_url = f'tcp://{host_name}:{_random_port}'
        # logger.info(f"Distributed URL: {dist_url}")
        
        hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(hostname)

        dist_url = f"tcp://{IPAddr}:{_random_port}"

        dist.launch(
            main,
            num_machines=_A.num_machines,
            num_gpus_per_machine=_A.num_gpus,
            machine_rank=_A.machine_rank,
            dist_url=dist_url,
            args=(_A,),
        )
        dist.dist.destroy_process_group()
