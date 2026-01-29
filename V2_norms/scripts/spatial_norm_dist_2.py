from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import Tensor
from omegaconf import OmegaConf
from loguru import logger
from tqdm import tqdm
from torch.cuda import amp
import seaborn as sns
import matplotlib.pyplot as plt

from hycoclip.config import LazyConfig, LazyFactory
from hycoclip.utils.checkpointing import CheckpointManager
from hycoclip.tokenizer import Tokenizer


parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA("--checkpoint-path", help="Path to checkpoint of a trained HyCoCLIP/MERU/CLIP model.")
_AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")
_AA("--dist-save-path", help="Path to save spatial norm distribution figure.")


def get_space_norm(x: Tensor):
    return torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True))


def main(_A: argparse.Namespace):
    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create evaluation and training config objects.
    _C_TRAIN = LazyConfig.load(_A.train_config)
    logger.info(OmegaConf.to_yaml(_C_TRAIN))

    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    dataloader = LazyFactory.build_dataloader(_C_TRAIN)
    tokenizer = Tokenizer()

    logger.info(f"Generating norm distribution for checkpoint in {_A.checkpoint_path}...")

    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(_A.checkpoint_path)
    model = model.eval()

    image_norms, box_image_norms = [], []
    text_norms, box_text_norms = [], []
    batches = 0

    hier_norms_0, hier_norms_1, hier_norms_2, hier_norms_3, hier_norms_4 = [], [], [], [], []

    for batch in tqdm(dataloader, desc=f"Generating representation norms"):

        with torch.inference_mode():
            tokens = tokenizer(batch["text"])
            box_tokens = tokenizer(batch["box_text"])
            
            hierarchy_tokens = [tokenizer(hier) for hier in batch["hierarchy"]]

            image_feats = model.encode_image(batch["image"].to(model.device), project=True)
            box_image_feats = model.encode_image(batch["box_image"].to(model.device), project=True)
            text_feats = model.encode_text(tokens, project=True)
            box_text_feats = model.encode_text(box_tokens, project=True)

            hierarchy_feats_0 = model.encode_text([hier[0] for hier in hierarchy_tokens], project=True)
            hierarchy_feats_1 = model.encode_text([hier[1] for hier in hierarchy_tokens], project=True)
            hierarchy_feats_2 = model.encode_text([hier[2] for hier in hierarchy_tokens], project=True)
            hierarchy_feats_3 = model.encode_text([hier[3] for hier in hierarchy_tokens], project=True)
            hierarchy_feats_4 = model.encode_text([hier[4] for hier in hierarchy_tokens], project=True)

            image_norms.append(get_space_norm(image_feats).to("cpu").detach().numpy())
            box_image_norms.append(get_space_norm(box_image_feats).to("cpu").detach().numpy())
            text_norms.append(get_space_norm(text_feats).to("cpu").detach().numpy())
            box_text_norms.append(get_space_norm(box_text_feats).to("cpu").detach().numpy())

            hier_norms_0.append(get_space_norm(hierarchy_feats_0).to("cpu").detach().numpy())
            hier_norms_1.append(get_space_norm(hierarchy_feats_1).to("cpu").detach().numpy())
            hier_norms_2.append(get_space_norm(hierarchy_feats_2).to("cpu").detach().numpy())
            hier_norms_3.append(get_space_norm(hierarchy_feats_3).to("cpu").detach().numpy())
            hier_norms_4.append(get_space_norm(hierarchy_feats_4).to("cpu").detach().numpy())

            batches += 1
            
            if batches > 167:      # Limit to 167 batches.
                break
    
    image_norms = np.concatenate(image_norms, axis=0)
    box_image_norms = np.concatenate(box_image_norms, axis=0)
    text_norms = np.concatenate(text_norms, axis=0)
    box_text_norms = np.concatenate(box_text_norms, axis=0)

    hier_norms_0 = np.concatenate(hier_norms_0, axis=0)
    hier_norms_1 = np.concatenate(hier_norms_1, axis=0)
    hier_norms_2 = np.concatenate(hier_norms_2, axis=0)
    hier_norms_3 = np.concatenate(hier_norms_3, axis=0)     
    hier_norms_4 = np.concatenate(hier_norms_4, axis=0)

    logger.info(f"Shape of norms: {image_norms.shape}, {box_image_norms.shape}, {text_norms.shape}, {box_text_norms.shape}")
    logger.info(f"Image norm, Mean: {image_norms.mean():.4f}, Min: {image_norms.min():.4f}, Max: {image_norms.max():.4f}")
    logger.info(f"Box image norm, Mean: {box_image_norms.mean():.4f}, Min: {box_image_norms.min():.4f}, Max: {box_image_norms.max():.4f}")
    logger.info(f"Text norm, Mean: {text_norms.mean():.4f}, Min: {text_norms.min():.4f}, Max: {text_norms.max():.4f}")
    logger.info(f"Box text norm, Mean: {box_text_norms.mean():.4f}, Min: {box_text_norms.min():.4f}, Max: {box_text_norms.max():.4f}")
  
    logger.info(f"Hierarchy level 0 (specific) norm, Mean: {hier_norms_0.mean():.4f}, Min: {hier_norms_0.min():.4f}, Max: {hier_norms_0.max():.4f}")
    logger.info(f"Hierarchy level 1 norm, Mean: {hier_norms_1.mean():.4f}, Min: {hier_norms_1.min():.4f}, Max: {hier_norms_1.max():.4f}")
    logger.info(f"Hierarchy level 2 norm, Mean: {hier_norms_2.mean():.4f}, Min: {hier_norms_2.min():.4f}, Max: {hier_norms_2.max():.4f}")
    logger.info(f"Hierarchy level 3 norm, Mean: {hier_norms_3.mean():.4f}, Min: {hier_norms_3.min():.4f}, Max: {hier_norms_3.max():.4f}")
    logger.info(f"Hierarchy level 4 (general) norm, Mean: {hier_norms_4.mean():.4f}, Min: {hier_norms_4.min():.4f}, Max: {hier_norms_4.max():.4f}")



    plt.figure(figsize=(10,4))
    sns.histplot(data=image_norms.squeeze(), bins='auto', stat="percent", kde=True, element="step", alpha=0.5, label='Images')
    sns.histplot(data=box_image_norms.squeeze(), bins='auto', stat="percent", kde=True, element="step", alpha=0.5, label='Image boxes')
    sns.histplot(data=text_norms.squeeze(), bins='auto', stat="percent", kde=True, element="step", alpha=0.5, label='Texts')
    sns.histplot(data=box_text_norms.squeeze(), bins='auto', stat="percent", kde=True, element="step", alpha=0.5, label='Text boxes')

    #sns.histplot(data=hier_norms_0.squeeze(), bins='auto', stat="percent", kde=True, element="step", alpha=0.5, label='H0 (specific)')
    sns.histplot(data=hier_norms_1.squeeze(), bins='auto', stat="percent", kde=True, element="step", alpha=0.5, label='H1')
    sns.histplot(data=hier_norms_2.squeeze(), bins='auto', stat="percent", kde=True, element="step", alpha=0.5, label='H2')
    sns.histplot(data=hier_norms_3.squeeze(), bins='auto', stat="percent", kde=True, element="step", alpha=0.5, label='H3')
    sns.histplot(data=hier_norms_4.squeeze(), bins='auto', stat="percent", kde=True, element="step", alpha=0.5, label='H4 (general)')


    plt.xlim(0.0)
    plt.xlabel((r'$\Vert \mathbf{\tilde{p}} \Vert$'))
    plt.ylabel('% of samples')
    plt.legend()
    
    plt.savefig(_A.dist_save_path, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
