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


def create_scatterplot(domain_words, domain_colors, domain_titles, domain_norms, save_path):

    plt.figure(figsize=(12, 6))

    for domain_id, words in domain_words.items():
        # Generate random norm values between 0.2 and 0.35
        norms = domain_norms[domain_id]
        plt.plot(words, norms, marker='o', linestyle='-', color=domain_colors[domain_id], label=f"Domain: {domain_titles[domain_id]}")

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Norm value")
    plt.title("Word Norms Across Domains")
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def create_scatterplot2(domain_words, domain_colors, domain_titles, domain_norms, save_path):

    plt.figure(figsize=(14, 6))

    # We'll spread the words for each domain horizontally with an offset
    x_base = 0
    x_ticks = []
    x_labels = []

    for domain_id, words in domain_words.items():
        # Generate random norm values
        #norms = [round(random.uniform(0.2, 0.35), 3) for _ in words]
        norms = domain_norms[domain_id]
        
        # Horizontal positions with small spacing for this domain
        x_positions = [x_base + i for i in range(len(words))]
        
        plt.plot(x_positions, norms, marker='o', linestyle='-', color=domain_colors[domain_id], label=f"Domain: {domain_titles[domain_id]}")
        
        # Record tick positions and labels
        x_ticks.extend(x_positions)
        x_labels.extend(words)
        
        # Update base for next domain (add extra space between domains)
        x_base += len(words) + 2

    plt.xticks(x_ticks, x_labels, rotation=45, ha="right")
    plt.ylabel("Embedding Distances")
    plt.title("Word Norms Across Domains")
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight", dpi=300)


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
    
    # Domains with 5 levels each
    domain_words = {
        1: ["entity", "thing", "object", "his new laptop", "his new silver MacBook Pro"],
        2: ["entity", "living being", "organism", "person", "my uncle Joe fixing his car"],
        3: ["entity", "concept", "theory", "scientific theory", "the theory of relativity"],
        4: ["entity", "organization", "company", "technology company", "Apple Inc."],
        5: ["entity", "occurrence", "event", "celebration", "my sister's wedding reception"]
    }

    domain_titles = {
        1: "Item",
        2: "Person",
        3: "Concept",
        4: "Group",
        5: "Event"
    }

    # Assign colors to each domain
    domain_colors = {
        1: "blue",
        2: "green",
        3: "orange",
        4: "red",
        5: "purple"
    }

    domain_norms = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
    }

    with torch.inference_mode():
        for key, word_list in domain_words.items():

            tokens = tokenizer(word_list)
            text_feats = model.encode_text(tokens, project=True)
            text_norms = get_space_norm(text_feats).to("cpu").detach().numpy()
            domain_norms[key] = np.concatenate(text_norms, axis=0)

            for i, word in enumerate(word_list):
                logger.info(f"Word: {word}, Norm: {text_norms[i]}")
            logger.info(f"\n")

    #create_scatterplot(domain_words, domain_colors, domain_titles, domain_norms, _A.dist_save_path+"_1")
    create_scatterplot2(domain_words, domain_colors, domain_titles, domain_norms, _A.dist_save_path)


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
