#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

# Modified from github.com/facebookresearch/meru

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

import hycoclip.utils.distributed as dist
from hycoclip import lorentz as L
from hycoclip.encoders.text_encoders import TransformerTextEncoder

from hycoclip.models2 import MERU

class HyCoCLIP_Re_Weight_DinContrastive(MERU):
    """
    Our HyCoCLIP_Re_Weight model, that modifies MERU to embed images, texts,
    their localized box and hierarchy information in a hyperbolic space.
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        contrast_weight: float = 1.0,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_hierarchies: bool = True,
    ):
        """
        Un-documented args are same as `HyCoCLIP`.

        Args:
            use_hierarchies: Whether to use caption hierarchies for training.
            cont_weights: Hyperparameters for hierarchical contrastive loss weights.

        """
        super().__init__(visual=visual,
                         textual=textual,
                         embed_dim=embed_dim,
                         curv_init=curv_init,
                         learn_curv=learn_curv,
                         entail_weight=entail_weight,
                         use_boxes=False, # NOTE: use-boxes is always set to false for the parent, otherwise the forward loop is being overwritten
                         pixel_mean=pixel_mean,
                         pixel_std=pixel_std)
        
        self.contrast_weight = contrast_weight

        assert use_hierarchies, "HyCoCLIP_Re_Weight requires caption hierarchies to function."

    def forward(
        self, images: torch.Tensor, box_images: torch.Tensor,
        tokens: list[torch.Tensor], box_tokens: list[torch.Tensor],
        hierarchy_tokens: list[list[torch.Tensor]], pairwise_scores: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            hierarchy_tokens: list of list of tensors, each containing hierarchy tokens
                for each phrase in the hierarchy.
        """

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        box_image_feats = self.encode_image(box_images, project=True)
        box_text_feats = self.encode_text(box_tokens, project=True)

        # compute hierarchy features
        hierarchy_feats_0 = self.encode_text([hier[0] for hier in hierarchy_tokens], project=True)
        hierarchy_feats_1 = self.encode_text([hier[1] for hier in hierarchy_tokens], project=True)
        hierarchy_feats_2 = self.encode_text([hier[2] for hier in hierarchy_tokens], project=True)
        hierarchy_feats_3 = self.encode_text([hier[3] for hier in hierarchy_tokens], project=True)
        hierarchy_feats_4 = self.encode_text([hier[4] for hier in hierarchy_tokens], project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)
        all_box_text_feats = dist.gather_across_processes(box_text_feats)
        all_box_image_feats = dist.gather_across_processes(box_image_feats)
        # all_hier_feats = dist.gather_across_processes(hierarchy_feats)

        all_hier_feats_0 = dist.gather_across_processes(hierarchy_feats_0)
        all_hier_feats_1 = dist.gather_across_processes(hierarchy_feats_1)
        all_hier_feats_2 = dist.gather_across_processes(hierarchy_feats_2)
        all_hier_feats_3 = dist.gather_across_processes(hierarchy_feats_3)
        all_hier_feats_4 = dist.gather_across_processes(hierarchy_feats_4)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)
        all_box_text_feats = torch.cat(all_box_text_feats, dim=0)
        all_box_image_feats = torch.cat(all_box_image_feats, dim=0)
        # shape: (batch_size * world_size, hierarchy_size(=5), embed_dim)
        # all_hier_feats = torch.cat(all_hier_feats, dim=0)
        all_hier_feats_0 = torch.cat(all_hier_feats_0, dim=0)
        all_hier_feats_1 = torch.cat(all_hier_feats_1, dim=0)
        all_hier_feats_2 = torch.cat(all_hier_feats_2, dim=0)
        all_hier_feats_3 = torch.cat(all_hier_feats_3, dim=0)
        all_hier_feats_4 = torch.cat(all_hier_feats_4, dim=0)

        # Compute all necessary loss components. We enclose the entire block with
        # autocast to force a higher floating point precision.
        with torch.autocast(self.device.type, dtype=torch.float32):
            # Compute logits for contrastive loss.
            image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
            text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)
            box_image_logits = -L.pairwise_dist(box_image_feats, all_text_feats, _curv)
            box_text_logits = -L.pairwise_dist(box_text_feats, all_image_feats, _curv)

            # using hyperbolic distance
            hier_logits_1 = L.elementwise_dist(hierarchy_feats_0, hierarchy_feats_1, _curv)
            hier_logits_2 = L.elementwise_dist(hierarchy_feats_1, hierarchy_feats_2, _curv)
            hier_logits_3 = L.elementwise_dist(hierarchy_feats_2, hierarchy_feats_3, _curv)
            hier_logits_4 = L.elementwise_dist(hierarchy_feats_3, hierarchy_feats_4, _curv)

            # mean elementwise distance between the hierarchy entries 
            mean_elementwise_dist = 0.25*(
                (hier_logits_1**2).mean() +
                (hier_logits_2**2).mean() +
                (hier_logits_3**2).mean() +
                (hier_logits_4**2).mean())
                
            # Compute cross entropy loss: we compute log probabilities and take the
            # diagonal elements as targets: image[i] should match text[i] in batch.
            # Shift the targets according to rank of GPU process (we assume that all
            # GPU processes have the same local batch size).
            batch_size = image_feats.shape[0]
            targets = torch.arange(batch_size, device=image_logits.device)
            targets = targets + batch_size * self._rank

            # Clamp temperature such that logits are not scaled more than 100x.
            # ln(100) = ~4.6052
            self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
            _scale = self.logit_scale.exp()

            # original hycoclip contrastive loss
            # Re Weight TOGETHER: 
            contrastive_loss = (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * text_logits, targets)
                + nn.functional.cross_entropy(_scale * box_image_logits, targets)
                + nn.functional.cross_entropy(_scale * box_text_logits, targets)

                # hierarchy contrast
                + mean_elementwise_dist
            )

            # Hyperbolic entailment loss: text should entail matching image.
            _angle = L.oxy_angle(text_feats, image_feats, _curv)
            _aperture = L.half_aperture(text_feats, _curv)

            _box_angle = L.oxy_angle(box_text_feats, box_image_feats, _curv)
            _box_aperture = L.half_aperture(box_text_feats, _curv)

            _cross_image_angle = L.oxy_angle(box_image_feats, image_feats, _curv)
            _box_image_aperture = L.half_aperture(box_image_feats, _curv)

            _cross_text_angle = L.oxy_angle(box_text_feats, text_feats, _curv)
            _box_text_aperture = L.half_aperture(box_text_feats, _curv)

            # angle chain between each consecutive element in the hierarchy
            hier_chain_angle_1 = L.oxy_angle(hierarchy_feats_1, hierarchy_feats_0, _curv)
            hier_chain_angle_2 = L.oxy_angle(hierarchy_feats_2, hierarchy_feats_1, _curv)
            hier_chain_angle_3 = L.oxy_angle(hierarchy_feats_3, hierarchy_feats_2, _curv)
            hier_chain_angle_4 = L.oxy_angle(hierarchy_feats_4, hierarchy_feats_3, _curv)

            # apertures of hierarchy entries
            hier_aperture_1 = L.half_aperture(hierarchy_feats_1, _curv)
            hier_aperture_2 = L.half_aperture(hierarchy_feats_2, _curv)
            hier_aperture_3 = L.half_aperture(hierarchy_feats_3, _curv)
            hier_aperture_4 = L.half_aperture(hierarchy_feats_4, _curv)

            # Hyperparameters for apertures
            _global_aperture_thresh = 0.7   # check = 1 ?
            _local_aperture_thresh = 1.2

            text_image_entailment_loss = torch.clamp(_angle - _global_aperture_thresh * _aperture, min=0).mean()
            box_text_image_entailment_loss = torch.clamp(_box_angle - _global_aperture_thresh * _box_aperture, min=0).mean()
            cross_image_entailment_loss = torch.clamp(_cross_image_angle - _local_aperture_thresh * _box_image_aperture, min=0).mean()
            cross_text_entailment_loss = torch.clamp(_cross_text_angle - _local_aperture_thresh * _box_text_aperture, min=0).mean()

            # Re_Weight: entailment chain using original entailment technique and global aperture threshold
            hier_entailment_loss_1 = torch.clamp(hier_chain_angle_1 - _global_aperture_thresh * hier_aperture_1, min=0).mean()
            hier_entailment_loss_2 = torch.clamp(hier_chain_angle_2 - _global_aperture_thresh * hier_aperture_2, min=0).mean()
            hier_entailment_loss_3 = torch.clamp(hier_chain_angle_3 - _global_aperture_thresh * hier_aperture_3, min=0).mean()
            hier_entailment_loss_4 = torch.clamp(hier_chain_angle_4 - _global_aperture_thresh * hier_aperture_4, min=0).mean()

            # new entailment loss that also includes re_weight part
            entailment_loss = 0.5 * (
                text_image_entailment_loss 
                + box_text_image_entailment_loss 
                + cross_image_entailment_loss 
                + cross_text_entailment_loss

                # Re_Weight: modulating the entailments with the pairwise scores
                + (pairwise_scores[0] * hier_entailment_loss_1).mean()
                + (pairwise_scores[1] * hier_entailment_loss_2).mean()
                + (pairwise_scores[2] * hier_entailment_loss_3).mean()
                + (pairwise_scores[3] * hier_entailment_loss_4).mean()
            )

            loss = self.contrast_weight * contrastive_loss + self.entail_weight * entailment_loss

        returnDict = {
            "loss": loss,
            "logging": {
                "loss": loss,
                "contrastive_loss": contrastive_loss,
                "entailment_loss": entailment_loss,
                "logit_scale": _scale,
                "curv": _curv,
            }
        }

        return returnDict
