from __future__ import annotations

import os
import sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from omegaconf import OmegaConf
from loguru import logger
from tqdm import tqdm

from hycoclip.config import LazyConfig, LazyFactory
from hycoclip.utils.checkpointing import CheckpointManager
from hycoclip.tokenizer import Tokenizer
from hycoclip import lorentz as L


parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA("--checkpoint-path", 
help="Path to checkpoint of a trained HyCoCLIP/MERU/CLIP model.")
_AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")
_AA("--embed-save-path", help="Path to save embeddings in .pkl format.")
_AA("--to-poincare", action='store_true', help="Whether to convert hyperboloid embeddings to Poincare ball.")
_AA("--reduce-dim", type=str, default=None, help="Dimensionality reduction method to use. Supported: horopca, cosne. If not set, no reduction is applied.")
_AA("--save-hier", help="Whether to save hierarchy embeddings as well.")

def create_hyperboloid_embed(x: Tensor, curv: float | Tensor = 1.0):
    """
    Compute the time dimension from spatial coordinates and return as Lorentzian N+1 dim vector.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, D+1)` giving full hyperboloid vector.
    """

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    x_full = torch.cat([x_time, x], dim=-1)
    return x_full


def get_space_norm(x: Tensor):
    return torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True))


def get_image_thumbnail(image: Tensor, size: int = 20):
    """
    Get a downsampled thumbnail of the image for visualization.

    Args:
        image: Tensor of shape `(3, H, W)` giving an image.
        size: Integer giving the height and width of the thumbnail.

    Returns:
        Tensor of shape `(size, size, 3)` giving the thumbnail.
    """
    thumbnail = F.interpolate(image, size=size, mode='bilinear', align_corners=False).squeeze(0).permute(0, 2, 3, 1)
    return thumbnail


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

    logger.info(f"Generating embeddings for checkpoint in {_A.checkpoint_path}...")

    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(_A.checkpoint_path)
    model = model.eval()

    all_image_feats, all_box_image_feats = [], []
    all_text_feats, all_box_text_feats = [], []
    # all_hier_feats = [[]]

    all_image_thumbnails, all_box_image_thumbnails = [], []
    all_texts, all_box_texts = [], []
    # all_hier_text = [[]]

    n_batches = 10

    for i, batch in zip(tqdm(range(n_batches), desc=f"Generating representations..."), dataloader):

        with torch.inference_mode():
            tokens = tokenizer(batch["text"])
            box_tokens = tokenizer(batch["box_text"])
            # hierarchy_tokens = [tokenizer(hier) for hier in batch["hierarchy"]]
            # logger.info(f"Shapes: {tokens.shape}, {box_tokens.shape}")#, {hierarchy_tokens.shape}")

            image_feats = model.encode_image(batch["image"].to(model.device), project=True)
            image_feats = create_hyperboloid_embed(image_feats, model.curv.exp())

            box_image_feats = model.encode_image(batch["box_image"].to(model.device), project=True)
            box_image_feats = create_hyperboloid_embed(box_image_feats, model.curv.exp())

            text_feats = model.encode_text(tokens, project=True)
            text_feats = create_hyperboloid_embed(text_feats, model.curv.exp())

            box_text_feats = model.encode_text(box_tokens, project=True)
            box_text_feats = create_hyperboloid_embed(box_text_feats, model.curv.exp())

            
            # return
            # hierarchy_feats_0 = self.encode_text([hier[0] for hier in hierarchy_tokens], project=True)

            all_image_feats.append(image_feats.to("cpu").detach())
            all_box_image_feats.append(box_image_feats.to("cpu").detach())
            all_text_feats.append(text_feats.to("cpu").detach())
            all_box_text_feats.append(box_text_feats.to("cpu").detach())
            
            all_box_texts.append(batch["box_text"])
            all_texts.append(batch["text"])
            all_image_thumbnails.append(get_image_thumbnail(batch["image"]).to("cpu").detach())
            all_box_image_thumbnails.append(get_image_thumbnail(batch["box_image"]).to("cpu").detach())
        
    all_image_feats = torch.concatenate(all_image_feats, axis=0)
    all_box_image_feats = torch.concatenate(all_box_image_feats, axis=0)
    all_text_feats = torch.concatenate(all_text_feats, axis=0)
    all_box_text_feats = torch.concatenate(all_box_text_feats, axis=0)
    
    all_image_thumbnails = torch.concatenate(all_image_thumbnails, axis=0).numpy()
    all_box_image_thumbnails = torch.concatenate(all_box_image_thumbnails, axis=0).numpy()
    all_texts = np.asarray(all_texts).flatten()
    all_box_texts = np.asarray(all_box_texts).flatten()
    
    if _A.to_poincare:
        logger.info("Converting hyperboloid embeddings to Poincare ball...")
        curv = model.curv.exp().to("cpu").detach()
        all_image_feats = L.lorentz_to_poincare(all_image_feats[..., 1:], curv)
        all_box_image_feats = L.lorentz_to_poincare(all_box_image_feats[..., 1:], curv)
        all_text_feats = L.lorentz_to_poincare(all_text_feats[..., 1:], curv)
        all_box_text_feats = L.lorentz_to_poincare(all_box_text_feats[..., 1:], curv)
        
    all_image_feats = all_image_feats.numpy()
    all_box_image_feats = all_box_image_feats.numpy()
    all_text_feats = all_text_feats.numpy()
    all_box_text_feats = all_box_text_feats.numpy()
    
    if _A.reduce_dim is not None:
        assert _A.to_poincare, "Dimensionality reduction is only supported for Poincare embeddings. Please set --to-poincare."
        logger.info(f"Reducing embeddings to 2D using {_A.reduce_dim}...")
        if _A.reduce_dim == "horopca":
            from hycoclip.utils.reduction import HoroPCAReduction
            reducer = HoroPCAReduction(dim=all_image_feats.shape[-1], n_components=2, lr=5e-2, max_steps=500, downsample=500)
        elif _A.reduce_dim == "cosne":
            from hycoclip.utils.reduction import COSNEReduction
            reducer = COSNEReduction(n_components=2)
        else:
            raise ValueError(f"Unknown reduction method {_A.reduce_dim}. Supported: horopca, cosne.")
        
        X = np.concatenate([all_image_feats, all_box_image_feats, all_text_feats, all_box_text_feats], axis=0)
        X = X.reshape(-1, X.shape[-1])
        X_reduced = reducer.fit_transform(X)
        n_samples = len(all_image_feats)
        all_image_feats = X_reduced[:n_samples]
        all_box_image_feats = X_reduced[n_samples:2*n_samples]
        all_text_feats = X_reduced[2*n_samples:3*n_samples]
        all_box_text_feats = X_reduced[3*n_samples:]
        logger.info("Dimensionality reduction complete.")
        
    embed_dict = {
        "image_feats": all_image_feats,
        "box_image_feats": all_box_image_feats,
        "text_feats": all_text_feats,
        "box_text_feats": all_box_text_feats,
        "image": all_image_thumbnails,
        "box_image": all_box_image_thumbnails,
        "text": all_texts,
        "box_text": all_box_texts
    }

    logger.info(f"Saving embeddings to {_A.embed_save_path}...")
    np.savez_compressed(_A.embed_save_path, **embed_dict)
    logger.info(f"Saved embeddings to {_A.embed_save_path}.")


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)