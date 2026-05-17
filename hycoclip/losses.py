import random

import torch
import torch.nn as nn
from hycoclip import lorentz as L

_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
_cast_dtype = torch.float32
_enable_autocast = torch.cuda.is_available() and _device_type == 'cuda'


def get_targets(batch_size, device, rank):
    """
    Returns the target indices for contrastive loss.
    The targets are shifted according to the rank of the GPU process.
    """
    targets = torch.arange(batch_size, device=device)
    targets = targets + batch_size * rank
    return targets

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def clip_loss(image_logits, text_logits, _rank, _scale, *args, **kwargs):
    targets = get_targets(image_logits.size(0), image_logits.device, _rank)

    loss = 0.5 * (
        nn.functional.cross_entropy(image_logits, targets)
        + nn.functional.cross_entropy(text_logits, targets)
    )
    
    return {
        "loss": loss,
        "logging": {"contrastive_loss": loss, "logit_scale": _scale},
    }

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def meru_loss(image_feats, text_feats, all_image_feats, all_text_feats, _curv, _rank, _scale, entail_weight=1.0):
    # Compute logits for contrastive loss.
    image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
    text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)

    # Compute cross entropy loss: we compute log probabilities and take the
    # diagonal elements as targets: image[i] should match text[i] in batch.
    # Shift the targets according to rank of GPU process (we assume that all
    # GPU processes have the same local batch size).
    targets = get_targets(image_logits.size(0), image_logits.device, _rank)

    contrastive_loss = 0.5 * (
        nn.functional.cross_entropy(_scale * image_logits, targets)
        + nn.functional.cross_entropy(_scale * text_logits, targets)
    )

    # Hyperbolic entailment loss: text should entail matching image.
    _angle = L.oxy_angle(text_feats, image_feats, _curv)
    _aperture = L.half_aperture(text_feats, _curv)

    entailment_loss = torch.clamp(_angle - _aperture, min=0).mean()

    loss = contrastive_loss
    if entail_weight > 0:
        loss = loss + entail_weight * entailment_loss

    return {
        "loss": loss,
        "logging": {
            "contrastive_loss": contrastive_loss,
            "entailment_loss": entailment_loss,
            "logit_scale": _scale,
            "curv": _curv,
        },
    }
    
# @torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
# def meru_with_boxes_loss(image_feats, text_feats, box_image_feats, box_text_feats, all_image_feats, all_text_feats,
#                          all_box_image_feats, all_box_text_feats, _curv, _rank, _scale, entail_weight=1.0):
#     text_image_loss = meru_loss(image_feats, text_feats, all_image_feats, all_text_feats, _curv, _rank, _scale, entail_weight)
#     box_text_image_loss = meru_loss(box_image_feats, box_text_feats, all_box_image_feats, all_box_text_feats, _curv, _rank, _scale, entail_weight)
    
#     loss = 0.5 * (text_image_loss["loss"] + box_text_image_loss["loss"])

#     return {
#         "loss": loss,
#         "logging": {
#             "contrastive_loss": 0.5 * (text_image_loss["logging"]["contrastive_loss"] + box_text_image_loss["logging"]["contrastive_loss"]),
#             "entailment_loss": 0.5 * (text_image_loss["logging"]["entailment_loss"] + box_text_image_loss["logging"]["entailment_loss"]),
#             "logit_scale": _scale,
#             "curv": _curv,
#         },
#     }

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def hycoclip_loss(image_feats, text_feats, box_image_feats, box_text_feats, all_image_feats, all_text_feats, _curv, _rank, _scale, entail_weight=1.0):
    """
    Computes the HyCoCLIP loss.
    """
    
    # Compute logits for contrastive loss.
    image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
    text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)
    box_image_logits = -L.pairwise_dist(box_image_feats, all_text_feats, _curv)
    box_text_logits = -L.pairwise_dist(box_text_feats, all_image_feats, _curv)

    # Compute cross entropy loss: we compute log probabilities and take the
    # diagonal elements as targets: image[i] should match text[i] in batch.
    # Shift the targets according to rank of GPU process (we assume that all
    # GPU processes have the same local batch size).
    targets = get_targets(image_logits.size(0), image_logits.device, _rank)

    contrastive_loss = 0.25 * (
        nn.functional.cross_entropy(_scale * image_logits, targets)
        + nn.functional.cross_entropy(_scale * text_logits, targets)
        + nn.functional.cross_entropy(_scale * box_image_logits, targets)
        + nn.functional.cross_entropy(_scale * box_text_logits, targets)
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

    # Hyperparameters for apertures
    _global_aperture_thresh = 0.7   # inter-modal
    _local_aperture_thresh = 1.2    # intra-modal

    text_image_entailment_loss = torch.clamp(_angle - _global_aperture_thresh * _aperture, min=0).mean()
    box_text_image_entailment_loss = torch.clamp(_box_angle - _global_aperture_thresh * _box_aperture, min=0).mean()
    cross_image_entailment_loss = torch.clamp(_cross_image_angle - _local_aperture_thresh * _box_image_aperture, min=0).mean()
    cross_text_entailment_loss = torch.clamp(_cross_text_angle - _local_aperture_thresh * _box_text_aperture, min=0).mean()

    entailment_loss = 0.5 * (
        text_image_entailment_loss 
        + box_text_image_entailment_loss 
        + cross_image_entailment_loss 
        + cross_text_entailment_loss
    )

    loss = contrastive_loss
    loss = loss + entail_weight * entailment_loss

    return {
        "loss": loss,
        "logging": {
            "contrastive_loss": contrastive_loss,
            "text_image_entailment_loss": text_image_entailment_loss,
            "box_text_image_entailment_loss": box_text_image_entailment_loss,
            "cross_image_entailment_loss": cross_image_entailment_loss,
            "cross_text_entailment_loss": cross_text_entailment_loss,
            "entailment_loss": entailment_loss,
            "logit_scale": _scale,
            "curv": _curv,
        },
    }


@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def hycoclip_loss_repulsion(image_feats, text_feats, box_image_feats, box_text_feats, all_image_feats, all_text_feats, _curv, _rank, _scale, entail_weight=1.0, repulsion_weight=0.1):
    """
    Computes the HyCoCLIP loss with repulsion term to push box embeddings apart.
    Repulsion loss: repulsion_weight * e^(-r) where r is the norm of box embeddings.
    """
    # Get standard HyCoCLIP loss
    base_loss = hycoclip_loss(image_feats, text_feats, box_image_feats, box_text_feats, 
                               all_image_feats, all_text_feats, _curv, _rank, _scale, entail_weight)
    
    # Compute repulsion loss: penalize when embeddings are close (small r)
    # r = norm of embeddings
    image_norms = torch.norm(image_feats, dim=-1)  # (batch_size,)
    text_norms = torch.norm(text_feats, dim=-1)    # (batch_size,)
    box_image_norms = torch.norm(box_image_feats, dim=-1)  # (batch_size,)
    box_text_norms = torch.norm(box_text_feats, dim=-1)    # (batch_size,)

    
    # Repulsion term: e^(-r) encourages large norms (dispersal in hyperbolic space)
    # When r is small, e^(-r) is large (high penalty)
    # When r is large, e^(-r) is small (low penalty)
    repulsion_loss = (torch.exp(-box_image_norms) + torch.exp(-box_text_norms) + torch.exp(-image_norms) + torch.exp(-text_norms)).mean()
    
    # Combine losses
    total_loss = base_loss["loss"] + repulsion_weight * repulsion_loss
    
    # Update logging with repulsion loss
    logging = base_loss["logging"].copy()
    logging["repulsion_loss"] = repulsion_loss
    
    return {
        "loss": total_loss,
        "logging": logging,
    }


@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def hycoclip_deep_loss(image_feats, text_feats, box_image_feats, box_text_feats, hierarchy_feats_list, hier_sample_type, all_image_feats, all_text_feats, _curv, _rank, _scale, entail_weight=1.0):
    """
    Computes the HyCoCLIP loss with extra hierarchical samples from Deep-GRIT.
    """
    
    if hier_sample_type == "ALL":
        # 0.25 --> 0.125 to account for the 4 extra hierarchical samples
        contrastive_loss_weight = 0.125
        # 0.5 --> 0.1667 to account for the 8 extra hierarchical samples
        entailment_loss_weight = 0.1667
    elif hier_sample_type == "SINGLE_RANDOM":
        # randomly sample an index to use as hierarchical sample for both contrastive and entailment loss, if using single random sampling
        random_hier_index = torch.randint(0, len(hierarchy_feats_list), (1, 1)).item()
        print("RANDOM HIER INDEX:", random_hier_index)
        # 0.25 --> 0.2 to account for the 1 extra hierarchical sample
        contrastive_loss_weight = 0.2
        # 0.5 --> 0.3333 to account for the 1 extra hierarchical sample
        entailment_loss_weight = 0.3333
    else:
        raise ValueError(f"Unknown hier_sample_type: {hier_sample_type}. Supported types are 'ALL' and 'SINGLE_RANDOM'.")


    # Compute logits for contrastive loss.
    image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
    text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)
    box_image_logits = -L.pairwise_dist(box_image_feats, all_text_feats, _curv)
    box_text_logits = -L.pairwise_dist(box_text_feats, all_image_feats, _curv)

    hier_logits = []
    for hierarchy_feats in hierarchy_feats_list:
        hier_logit = -L.pairwise_dist(hierarchy_feats, all_image_feats, _curv)
        hier_logits.append(hier_logit)

    # Compute cross entropy loss: we compute log probabilities and take the
    # diagonal elements as targets: image[i] should match text[i] in batch.
    # Shift the targets according to rank of GPU process (we assume that all
    # GPU processes have the same local batch size).
    targets = get_targets(image_logits.size(0), image_logits.device, _rank)


    if hier_sample_type == "ALL":
        # using all hierarchical samples with equal weighting
        hierarchical_contrastive_loss = sum(nn.functional.cross_entropy(_scale * hier_logit, targets) for hier_logit in hier_logits)
    elif hier_sample_type == "SINGLE_RANDOM":
        # randomly sample a single hierarchical sample
        hierarchical_contrastive_loss = nn.functional.cross_entropy(_scale * hier_logits[random_hier_index], targets)

    # 0.25 --> contrastive_loss_weight as defined above
    contrastive_loss = contrastive_loss_weight * (
        nn.functional.cross_entropy(_scale * image_logits, targets)
        + nn.functional.cross_entropy(_scale * text_logits, targets)
        + nn.functional.cross_entropy(_scale * box_image_logits, targets)
        + nn.functional.cross_entropy(_scale * box_text_logits, targets)
        + hierarchical_contrastive_loss
    )

    # Hyperbolic entailment loss: text should entail matching image.
    _angle = L.oxy_angle(text_feats, image_feats, _curv)
    _aperture = L.half_aperture(text_feats, _curv)

    _box_angle = L.oxy_angle(box_text_feats, box_image_feats, _curv)
    _box_text_aperture = L.half_aperture(box_text_feats, _curv)

    _cross_image_angle = L.oxy_angle(box_image_feats, image_feats, _curv)
    _box_image_aperture = L.half_aperture(box_image_feats, _curv)

    _cross_text_angle = L.oxy_angle(box_text_feats, text_feats, _curv)
    # _box_text_aperture = L.half_aperture(box_text_feats, _curv)


    _hier_angles = []
    _hier_apertures = []
    _hier_cross_text_angles = []
    for hierarchy_feats in hierarchy_feats_list:
        # compute angle between hierarchy term and box image
        hier_angle = L.oxy_angle(hierarchy_feats, box_image_feats, _curv)
        # aperture of hierarchy term
        hier_aperture = L.half_aperture(hierarchy_feats, _curv)
        # angle between hierarchy term and text
        hier_cross_text_angle = L.oxy_angle(hierarchy_feats, text_feats, _curv)

        _hier_angles.append(hier_angle)
        _hier_apertures.append(hier_aperture)
        _hier_cross_text_angles.append(hier_cross_text_angle)


    # Hyperparameters for apertures
    _global_aperture_thresh = 0.7   # inter-modal
    _local_aperture_thresh = 1.2    # intra-modal

    text_image_entailment_loss = torch.clamp(_angle - _global_aperture_thresh * _aperture, min=0).mean()
    box_text_image_entailment_loss = torch.clamp(_box_angle - _global_aperture_thresh * _box_text_aperture, min=0).mean()
    cross_image_entailment_loss = torch.clamp(_cross_image_angle - _local_aperture_thresh * _box_image_aperture, min=0).mean()
    cross_text_entailment_loss = torch.clamp(_cross_text_angle - _local_aperture_thresh * _box_text_aperture, min=0).mean()


    # every hier element should entail the box image
    hier_box_image_entailment_loss = [torch.clamp(hier_angle - _global_aperture_thresh * hier_aperture, min=0).mean()
                                      for hier_angle, hier_aperture in zip(_hier_angles, _hier_apertures)]
    
    # every hier element should entail the text
    hier_cross_text_entailment_loss = [torch.clamp(hier_cross_text_angle - _local_aperture_thresh * hier_aperture, min=0).mean()
                                       for hier_cross_text_angle, hier_aperture in zip(_hier_cross_text_angles, _hier_apertures)]

    # using all hierarchical samples
    if hier_sample_type == "ALL":
        # every hier element should entail the box image and every hier element should entail the text
        hierarchical_entailment_loss = sum(loss_term for loss_term in hier_box_image_entailment_loss) + sum(loss_term for loss_term in hier_cross_text_entailment_loss)
    elif hier_sample_type == "SINGLE_RANDOM":
        # randomly sample a single hierarchical sample
        hierarchical_entailment_loss = hier_box_image_entailment_loss[random_hier_index] + hier_cross_text_entailment_loss[random_hier_index]

    # 0.5 --> entailment_loss_weight as defined above
    entailment_loss = entailment_loss_weight * (
        text_image_entailment_loss 
        + box_text_image_entailment_loss 
        + cross_image_entailment_loss 
        + cross_text_entailment_loss

        + hierarchical_entailment_loss
    )

    loss = contrastive_loss
    loss = loss + entail_weight * entailment_loss

    return {
        "loss": loss,
        "logging": {
            "contrastive_loss": contrastive_loss,
            "text_image_entailment_loss": text_image_entailment_loss,
            "box_text_image_entailment_loss": box_text_image_entailment_loss,
            "cross_image_entailment_loss": cross_image_entailment_loss,
            "cross_text_entailment_loss": cross_text_entailment_loss,
            "entailment_loss": entailment_loss,
            "logit_scale": _scale,
            "curv": _curv,
        },
    }


@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def hyco_reweight_loss(image_feats, text_feats, box_image_feats, box_text_feats, all_image_feats, all_text_feats, hierarchy_feats, pairwise_scores, _curv, _rank, _scale, entail_weight=1.0, contrast_weight=1.0):
    """
    Computes the HyCoCLIP ReWeight loss.
    """
    
    # Compute logits for contrastive loss.
    image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
    text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)
    box_image_logits = -L.pairwise_dist(box_image_feats, all_text_feats, _curv)
    box_text_logits = -L.pairwise_dist(box_text_feats, all_image_feats, _curv)

    # using hyperbolic distance
    hier_logits_1 = L.elementwise_dist(box_text_feats, hierarchy_feats[0], _curv)
    hier_logits_2 = L.elementwise_dist(hierarchy_feats[0], hierarchy_feats[1], _curv)
    hier_logits_3 = L.elementwise_dist(hierarchy_feats[1], hierarchy_feats[2], _curv)
    hier_logits_4 = L.elementwise_dist(hierarchy_feats[2], hierarchy_feats[3], _curv)

    # mean elementwise distance between the hierarchy entries 
    mean_elementwise_dist = 0.25 * (
        (hier_logits_1**2).mean() +
        (hier_logits_2**2).mean() +
        (hier_logits_3**2).mean() +
        (hier_logits_4**2).mean()
    )

    # Compute cross entropy loss: we compute log probabilities and take the
    # diagonal elements as targets: image[i] should match text[i] in batch.
    # Shift the targets according to rank of GPU process (we assume that all
    # GPU processes have the same local batch size).
    targets = get_targets(image_logits.size(0), image_logits.device, _rank)

    contrastive_loss = (
        0.25*(
            nn.functional.cross_entropy(_scale * image_logits, targets)
            + nn.functional.cross_entropy(_scale * text_logits, targets)
            + nn.functional.cross_entropy(_scale * box_image_logits, targets)
            + nn.functional.cross_entropy(_scale * box_text_logits, targets)
        )
        # hierarchy contrast
        + (0.5 * mean_elementwise_dist)
    )   

    # tried min_radius=0.05 but also didnt include it in hierarchy half apertures
    # might have been a too harsh decrease from 0.1, try 0.75

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
    hier_chain_angle_1 = L.oxy_angle(hierarchy_feats[0], box_text_feats, _curv)
    hier_chain_angle_2 = L.oxy_angle(hierarchy_feats[1], hierarchy_feats[0], _curv)
    hier_chain_angle_3 = L.oxy_angle(hierarchy_feats[2], hierarchy_feats[1], _curv)
    hier_chain_angle_4 = L.oxy_angle(hierarchy_feats[3], hierarchy_feats[2], _curv)

    # apertures of hierarchy entries
    hier_aperture_1 = L.half_aperture(hierarchy_feats[0], _curv)
    hier_aperture_2 = L.half_aperture(hierarchy_feats[1], _curv)
    hier_aperture_3 = L.half_aperture(hierarchy_feats[2], _curv)
    hier_aperture_4 = L.half_aperture(hierarchy_feats[3], _curv)

    # Hyperparameters for apertures
    _global_aperture_thresh = 0.7   # inter-modal
    _local_aperture_thresh = 1.2    # intra-modal

    text_image_entailment_loss = torch.clamp(_angle - _global_aperture_thresh * _aperture, min=0).mean()
    box_text_image_entailment_loss = torch.clamp(_box_angle - _global_aperture_thresh * _box_aperture, min=0).mean()
    cross_image_entailment_loss = torch.clamp(_cross_image_angle - _local_aperture_thresh * _box_image_aperture, min=0).mean()
    cross_text_entailment_loss = torch.clamp(_cross_text_angle - _local_aperture_thresh * _box_text_aperture, min=0).mean()

    # Re_Weight: entailment chain using original entailment technique and global aperture threshold
    hier_entailment_loss_1 = torch.clamp(hier_chain_angle_1 - _global_aperture_thresh * hier_aperture_1, min=0).mean()
    hier_entailment_loss_2 = torch.clamp(hier_chain_angle_2 - _global_aperture_thresh * hier_aperture_2, min=0).mean()
    hier_entailment_loss_3 = torch.clamp(hier_chain_angle_3 - _global_aperture_thresh * hier_aperture_3, min=0).mean()
    hier_entailment_loss_4 = torch.clamp(hier_chain_angle_4 - _global_aperture_thresh * hier_aperture_4, min=0).mean()


    entailment_loss = (
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

    # loss = (contrast_weight * contrastive_loss) + (entail_weight * entailment_loss)
    loss = contrastive_loss + (entail_weight * entailment_loss)

    return {
        "loss": loss,
        "logging": {
            "contrastive_loss": contrastive_loss,
            "text_image_entailment_loss": text_image_entailment_loss,
            "box_text_image_entailment_loss": box_text_image_entailment_loss,
            "cross_image_entailment_loss": cross_image_entailment_loss,
            "cross_text_entailment_loss": cross_text_entailment_loss,
            "entailment_loss": entailment_loss,
            "logit_scale": _scale,
            "curv": _curv,
        },
    }


"""
-----------------------------------------------------------
Added for quick testing on smaller values of K
"""
@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def hyco_reweight_loss_slightly_smaller_K(image_feats, text_feats, box_image_feats, box_text_feats, all_image_feats, all_text_feats, hierarchy_feats, pairwise_scores, _curv, _rank, _scale, entail_weight=1.0, contrast_weight=1.0):
    """
    Computes the HyCoCLIP ReWeight loss with K = 0.05 instead of 0.1 when calculating the half aperture
    Otherwise it is the same as hyco_reweight_loss
    """

    # instead of the original 0.1, K = 0.09
    smaller_min_radius = 0.09
    
    # Compute logits for contrastive loss.
    image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
    text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)
    box_image_logits = -L.pairwise_dist(box_image_feats, all_text_feats, _curv)
    box_text_logits = -L.pairwise_dist(box_text_feats, all_image_feats, _curv)

    # using hyperbolic distance
    hier_logits_1 = L.elementwise_dist(box_text_feats, hierarchy_feats[0], _curv)
    hier_logits_2 = L.elementwise_dist(hierarchy_feats[0], hierarchy_feats[1], _curv)
    hier_logits_3 = L.elementwise_dist(hierarchy_feats[1], hierarchy_feats[2], _curv)
    hier_logits_4 = L.elementwise_dist(hierarchy_feats[2], hierarchy_feats[3], _curv)

    # mean elementwise distance between the hierarchy entries 
    mean_elementwise_dist = 0.25 * (
        (hier_logits_1**2).mean() +
        (hier_logits_2**2).mean() +
        (hier_logits_3**2).mean() +
        (hier_logits_4**2).mean()
    )

    # Compute cross entropy loss: we compute log probabilities and take the
    # diagonal elements as targets: image[i] should match text[i] in batch.
    # Shift the targets according to rank of GPU process (we assume that all
    # GPU processes have the same local batch size).
    targets = get_targets(image_logits.size(0), image_logits.device, _rank)

    contrastive_loss = (
        0.25*(
            nn.functional.cross_entropy(_scale * image_logits, targets)
            + nn.functional.cross_entropy(_scale * text_logits, targets)
            + nn.functional.cross_entropy(_scale * box_image_logits, targets)
            + nn.functional.cross_entropy(_scale * box_text_logits, targets)
        )
        # hierarchy contrast
        + (0.5 * mean_elementwise_dist)
    )   

    # Hyperbolic entailment loss: text should entail matching image.
    _angle = L.oxy_angle(text_feats, image_feats, _curv)
    _aperture = L.half_aperture(text_feats, _curv, min_radius=smaller_min_radius)

    _box_angle = L.oxy_angle(box_text_feats, box_image_feats, _curv)
    _box_aperture = L.half_aperture(box_text_feats, _curv, min_radius=smaller_min_radius)

    _cross_image_angle = L.oxy_angle(box_image_feats, image_feats, _curv)
    _box_image_aperture = L.half_aperture(box_image_feats, _curv, min_radius=smaller_min_radius)

    _cross_text_angle = L.oxy_angle(box_text_feats, text_feats, _curv)
    _box_text_aperture = L.half_aperture(box_text_feats, _curv, min_radius=smaller_min_radius)

    # angle chain between each consecutive element in the hierarchy
    hier_chain_angle_1 = L.oxy_angle(hierarchy_feats[0], box_text_feats, _curv)
    hier_chain_angle_2 = L.oxy_angle(hierarchy_feats[1], hierarchy_feats[0], _curv)
    hier_chain_angle_3 = L.oxy_angle(hierarchy_feats[2], hierarchy_feats[1], _curv)
    hier_chain_angle_4 = L.oxy_angle(hierarchy_feats[3], hierarchy_feats[2], _curv)

    # apertures of hierarchy entries
    hier_aperture_1 = L.half_aperture(hierarchy_feats[0], _curv, min_radius=smaller_min_radius)
    hier_aperture_2 = L.half_aperture(hierarchy_feats[1], _curv, min_radius=smaller_min_radius)
    hier_aperture_3 = L.half_aperture(hierarchy_feats[2], _curv, min_radius=smaller_min_radius)
    hier_aperture_4 = L.half_aperture(hierarchy_feats[3], _curv, min_radius=smaller_min_radius)

    # Hyperparameters for apertures
    _global_aperture_thresh = 0.7   # inter-modal
    _local_aperture_thresh = 1.2    # intra-modal

    text_image_entailment_loss = torch.clamp(_angle - _global_aperture_thresh * _aperture, min=0).mean()
    box_text_image_entailment_loss = torch.clamp(_box_angle - _global_aperture_thresh * _box_aperture, min=0).mean()
    cross_image_entailment_loss = torch.clamp(_cross_image_angle - _local_aperture_thresh * _box_image_aperture, min=0).mean()
    cross_text_entailment_loss = torch.clamp(_cross_text_angle - _local_aperture_thresh * _box_text_aperture, min=0).mean()

    # Re_Weight: entailment chain using original entailment technique and global aperture threshold
    hier_entailment_loss_1 = torch.clamp(hier_chain_angle_1 - _global_aperture_thresh * hier_aperture_1, min=0).mean()
    hier_entailment_loss_2 = torch.clamp(hier_chain_angle_2 - _global_aperture_thresh * hier_aperture_2, min=0).mean()
    hier_entailment_loss_3 = torch.clamp(hier_chain_angle_3 - _global_aperture_thresh * hier_aperture_3, min=0).mean()
    hier_entailment_loss_4 = torch.clamp(hier_chain_angle_4 - _global_aperture_thresh * hier_aperture_4, min=0).mean()


    entailment_loss = (
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

    # loss = (contrast_weight * contrastive_loss) + (entail_weight * entailment_loss)
    loss = contrastive_loss + (entail_weight * entailment_loss)

    return {
        "loss": loss,
        "logging": {
            "contrastive_loss": contrastive_loss,
            "text_image_entailment_loss": text_image_entailment_loss,
            "box_text_image_entailment_loss": box_text_image_entailment_loss,
            "cross_image_entailment_loss": cross_image_entailment_loss,
            "cross_text_entailment_loss": cross_text_entailment_loss,
            "entailment_loss": entailment_loss,
            "logit_scale": _scale,
            "curv": _curv,
        },
    }

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def hyco_reweight_loss_very_small_K(image_feats, text_feats, box_image_feats, box_text_feats, all_image_feats, all_text_feats, hierarchy_feats, pairwise_scores, _curv, _rank, _scale, entail_weight=1.0, contrast_weight=1.0):
    """
    Computes the HyCoCLIP ReWeight loss with K = 0.05 instead of 0.1 when calculating the half aperture
    Otherwise it is the same as hyco_reweight_loss
    """

    # instead of the original 0.1, K = 0.09
    smaller_min_radius = 0.075
    
    # Compute logits for contrastive loss.
    image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
    text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)
    box_image_logits = -L.pairwise_dist(box_image_feats, all_text_feats, _curv)
    box_text_logits = -L.pairwise_dist(box_text_feats, all_image_feats, _curv)

    # using hyperbolic distance
    hier_logits_1 = L.elementwise_dist(box_text_feats, hierarchy_feats[0], _curv)
    hier_logits_2 = L.elementwise_dist(hierarchy_feats[0], hierarchy_feats[1], _curv)
    hier_logits_3 = L.elementwise_dist(hierarchy_feats[1], hierarchy_feats[2], _curv)
    hier_logits_4 = L.elementwise_dist(hierarchy_feats[2], hierarchy_feats[3], _curv)

    # mean elementwise distance between the hierarchy entries 
    mean_elementwise_dist = 0.25 * (
        (hier_logits_1**2).mean() +
        (hier_logits_2**2).mean() +
        (hier_logits_3**2).mean() +
        (hier_logits_4**2).mean()
    )

    # Compute cross entropy loss: we compute log probabilities and take the
    # diagonal elements as targets: image[i] should match text[i] in batch.
    # Shift the targets according to rank of GPU process (we assume that all
    # GPU processes have the same local batch size).
    targets = get_targets(image_logits.size(0), image_logits.device, _rank)

    contrastive_loss = (
        0.25*(
            nn.functional.cross_entropy(_scale * image_logits, targets)
            + nn.functional.cross_entropy(_scale * text_logits, targets)
            + nn.functional.cross_entropy(_scale * box_image_logits, targets)
            + nn.functional.cross_entropy(_scale * box_text_logits, targets)
        )
        # hierarchy contrast
        + (0.5 * mean_elementwise_dist)
    )   

    # Hyperbolic entailment loss: text should entail matching image.
    _angle = L.oxy_angle(text_feats, image_feats, _curv)
    _aperture = L.half_aperture(text_feats, _curv, min_radius=smaller_min_radius)

    _box_angle = L.oxy_angle(box_text_feats, box_image_feats, _curv)
    _box_aperture = L.half_aperture(box_text_feats, _curv, min_radius=smaller_min_radius)

    _cross_image_angle = L.oxy_angle(box_image_feats, image_feats, _curv)
    _box_image_aperture = L.half_aperture(box_image_feats, _curv, min_radius=smaller_min_radius)

    _cross_text_angle = L.oxy_angle(box_text_feats, text_feats, _curv)
    _box_text_aperture = L.half_aperture(box_text_feats, _curv, min_radius=smaller_min_radius)

    # angle chain between each consecutive element in the hierarchy
    hier_chain_angle_1 = L.oxy_angle(hierarchy_feats[0], box_text_feats, _curv)
    hier_chain_angle_2 = L.oxy_angle(hierarchy_feats[1], hierarchy_feats[0], _curv)
    hier_chain_angle_3 = L.oxy_angle(hierarchy_feats[2], hierarchy_feats[1], _curv)
    hier_chain_angle_4 = L.oxy_angle(hierarchy_feats[3], hierarchy_feats[2], _curv)

    # apertures of hierarchy entries
    hier_aperture_1 = L.half_aperture(hierarchy_feats[0], _curv, min_radius=smaller_min_radius)
    hier_aperture_2 = L.half_aperture(hierarchy_feats[1], _curv, min_radius=smaller_min_radius)
    hier_aperture_3 = L.half_aperture(hierarchy_feats[2], _curv, min_radius=smaller_min_radius)
    hier_aperture_4 = L.half_aperture(hierarchy_feats[3], _curv, min_radius=smaller_min_radius)

    # Hyperparameters for apertures
    _global_aperture_thresh = 0.7   # inter-modal
    _local_aperture_thresh = 1.2    # intra-modal

    text_image_entailment_loss = torch.clamp(_angle - _global_aperture_thresh * _aperture, min=0).mean()
    box_text_image_entailment_loss = torch.clamp(_box_angle - _global_aperture_thresh * _box_aperture, min=0).mean()
    cross_image_entailment_loss = torch.clamp(_cross_image_angle - _local_aperture_thresh * _box_image_aperture, min=0).mean()
    cross_text_entailment_loss = torch.clamp(_cross_text_angle - _local_aperture_thresh * _box_text_aperture, min=0).mean()

    # Re_Weight: entailment chain using original entailment technique and global aperture threshold
    hier_entailment_loss_1 = torch.clamp(hier_chain_angle_1 - _global_aperture_thresh * hier_aperture_1, min=0).mean()
    hier_entailment_loss_2 = torch.clamp(hier_chain_angle_2 - _global_aperture_thresh * hier_aperture_2, min=0).mean()
    hier_entailment_loss_3 = torch.clamp(hier_chain_angle_3 - _global_aperture_thresh * hier_aperture_3, min=0).mean()
    hier_entailment_loss_4 = torch.clamp(hier_chain_angle_4 - _global_aperture_thresh * hier_aperture_4, min=0).mean()


    entailment_loss = (
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

    # loss = (contrast_weight * contrastive_loss) + (entail_weight * entailment_loss)
    loss = contrastive_loss + (entail_weight * entailment_loss)

    return {
        "loss": loss,
        "logging": {
            "contrastive_loss": contrastive_loss,
            "text_image_entailment_loss": text_image_entailment_loss,
            "box_text_image_entailment_loss": box_text_image_entailment_loss,
            "cross_image_entailment_loss": cross_image_entailment_loss,
            "cross_text_entailment_loss": cross_text_entailment_loss,
            "entailment_loss": entailment_loss,
            "logit_scale": _scale,
            "curv": _curv,
        },
    }
"""
Added for quick testing on smaller values of K
-----------------------------------------------------------
"""


@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def accept_the_modality_gap_loss(image_feats, text_feats, all_image_feats, all_text_feats, _curv, _rank, _scale, entail_weight=1.0):
    """
    Computes the Accept the Modality Gap loss.
    """
    
    text_logits = -L.pairwise_oxy_angle(text_feats, all_image_feats, _curv)
    image_logits = L.pairwise_oxy_angle(image_feats, all_text_feats, _curv)
    
    targets = get_targets(image_logits.size(0), image_logits.device, _rank)
    
    contrastive_loss = 0.5*(
        nn.functional.cross_entropy(_scale * image_logits, targets) + 
        nn.functional.cross_entropy(_scale * text_logits, targets)
    )
    
    # # Midpoint regularization (absent in the official implementation but described in the paper as part of the proposed method)
    # text_centroid = L.einstein_midpoint(text_feats, _curv)
    # image_centroid = L.einstein_midpoint(image_feats, _curv)
    
    # _sqrt_curv = torch.sqrt(_curv)
    # q, p = 1.1, 1.3 # these values are set arbitrarily, as the official implementation does not specify them nor the paper does
    # # also, according to the paper, this is component-wise
    # text_thr = torch.acosh(torch.clamp(_curv * q, min=1.0+1e-6)) / _sqrt_curv
    # image_thr = torch.acosh(torch.clamp(_curv * p, min=1.0+1e-6)) / _sqrt_curv
    # midpoint_text_reg = torch.sum((text_centroid - text_thr)**2, dim=-1).mean()
    # midpoint_image_reg = torch.sum((image_centroid - image_thr)**2, dim=-1).mean()

    loss = contrastive_loss #+ entail_weight * (midpoint_text_reg + midpoint_image_reg)

    return {
        "loss": loss,
        "logging": {
            "contrastive_loss": contrastive_loss,
            # "midpoint_text_reg": midpoint_text_reg,
            # "midpoint_image_reg": midpoint_image_reg,
            "logit_scale": _scale,
            "curv": _curv,
        },
    }
    
#### Chord loss, or Death and all its friends

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def meru_chord_loss(image_feats, text_feats, all_image_feats, all_text_feats, _curv, _rank, _scale, entail_weight=1.0):
    """
    Computes the CHOrd loss.
    """
    # Compute logits for contrastive loss.
    image_to_text_logits = -L.pairwise_chord_length(all_text_feats, image_feats, _curv).permute(1,0) # (local_batch_size, global_batch_size) # all_image_feats
    text_to_image_logits = -L.pairwise_chord_length(text_feats, all_image_feats, _curv) # (local_batch_size, global_batch_size) # all_text_feats

    # Compute cross entropy loss: we compute log probabilities and take the
    # diagonal elements as targets: image[i] should match text[i] in batch.
    # Shift the targets according to rank of GPU process (we assume that all
    # GPU processes have the same local batch size).
    global_targets = get_targets(text_to_image_logits.size(0), text_to_image_logits.device, _rank)
    
    image_contr_entailment_loss = nn.functional.cross_entropy(_scale * image_to_text_logits, global_targets) # global
    text_contr_entailment_loss = nn.functional.cross_entropy(_scale * text_to_image_logits, global_targets) # global

    contrastive_loss = 0.50 * (
        image_contr_entailment_loss
        + text_contr_entailment_loss
    )

    return {
        "loss": contrastive_loss,
        "logging": {
            "contrastive_loss": contrastive_loss,
            "image_contr_entailment_loss": image_contr_entailment_loss,
            "text_contr_entailment_loss": text_contr_entailment_loss,
            "logit_scale": _scale,
            "curv": _curv,
        },
    }
    
@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def chordclip_loss(image_feats, text_feats, box_image_feats, box_text_feats, all_image_feats, all_text_feats, _curv, _rank, _scale, entail_weight=1.0):
    """
    Computes the CHOrd loss.
    """
    # Compute logits for contrastive loss
    image_to_text_logits = -L.safe_pairwise_chord_length(all_text_feats, image_feats, _curv).permute(1,0) # (local_batch_size, global_batch_size) # all_image_feats
    text_to_image_logits = -L.safe_pairwise_chord_length(text_feats, all_image_feats, _curv) # (local_batch_size, global_batch_size) # all_text_feats
    text_to_box_image_logits = -L.safe_pairwise_chord_length(all_text_feats, box_image_feats, _curv).permute(1,0) # (local_batch_size, global_batch_size) # all_text_feats
    box_text_to_image_logits = -L.safe_pairwise_chord_length(box_text_feats, all_image_feats, _curv) # (local_batch_size, global_batch_size) # all_image_feats
    box_text_to_box_image_logits = -L.safe_pairwise_chord_length(box_text_feats, box_image_feats, _curv).permute(1,0) # (local_batch_size, local_batch_size)

    # Compute cross entropy loss: we compute log probabilities and take the
    # diagonal elements as targets: image[i] should match text[i] in batch.
    # Shift the targets according to rank of GPU process (we assume that all
    # GPU processes have the same local batch size).
    global_targets = get_targets(text_to_image_logits.size(0), text_to_image_logits.device, _rank)
    local_targets = get_targets(text_to_image_logits.size(0), text_to_image_logits.device, 0) # rank set to 0 so that targets are in [0, local_batch_size-1]
    
    image_contr_entailment_loss = nn.functional.cross_entropy(_scale * image_to_text_logits, global_targets) # global
    text_contr_entailment_loss = nn.functional.cross_entropy(_scale * text_to_image_logits, global_targets) # global
    text_to_box_image_contr_entailment_loss = nn.functional.cross_entropy(_scale * text_to_box_image_logits, global_targets) # global
    box_text_to_image_contr_entailment_loss = nn.functional.cross_entropy(_scale * box_text_to_image_logits, global_targets) # global
    box_text_to_box_image_contr_entailment_loss = nn.functional.cross_entropy(_scale * box_text_to_box_image_logits, local_targets) # local

    # Compute aperture losses
    _cross_text_angle = L.oxy_angle(box_text_feats, text_feats, _curv)
    _box_text_aperture = L.half_aperture(box_text_feats, _curv)

    # Hyperparameters for apertures
    _local_aperture_thresh = 1.2    # intra-modal
    
    cross_text_entailment_loss = torch.clamp(_cross_text_angle - _local_aperture_thresh * _box_text_aperture, min=0).mean()
    
    contrastive_loss = 0.20 * (
        image_contr_entailment_loss
        + text_contr_entailment_loss
        + text_to_box_image_contr_entailment_loss
        + box_text_to_image_contr_entailment_loss
        + box_text_to_box_image_contr_entailment_loss
    )
    
    loss = contrastive_loss + entail_weight * cross_text_entailment_loss
    return {
        "loss": loss,
        "logging": {
            "contrastive_loss": contrastive_loss,
            "image_contr_entailment_loss": image_contr_entailment_loss,
            "text_contr_entailment_loss": text_contr_entailment_loss,
            "text_to_box_image_contr_entailment_loss": text_to_box_image_contr_entailment_loss,
            "box_text_to_image_contr_entailment_loss": box_text_to_image_contr_entailment_loss,
            "box_text_to_box_image_contr_entailment_loss": box_text_to_box_image_contr_entailment_loss,
            "cross_text_entailment_loss": cross_text_entailment_loss,
            "logit_scale": _scale,
            "curv": _curv,
        },
    }