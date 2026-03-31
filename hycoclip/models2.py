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
from hycoclip import losses
from hycoclip.encoders.text_encoders import TransformerTextEncoder

# from fast.projects.amsterdam.InternVL.internvl_chat_llava.llava.model.multimodal_encoder.eva_clip.modeling_evaclip import contrastive_loss
from enum import Enum

class HierarchySampleType(Enum):
    ALL = "ALL"
    SINGLE_RANDOM = "SINGLE_RANDOM"


def truncate_tokens(tokens: torch.Tensor, context_length: int) -> torch.Tensor:
    # Truncate tokens that are longer than context_length:
    eot_token = tokens[-1]
    tokens = tokens[:context_length]
    tokens[-1] = eot_token
    return tokens


class CLIPBaseline(nn.Module):
    """
    Re-implementation of the CLIP model that uses an image-text contrastive
    loss as a training objective and embeds images and text in a Euclidean space.

    Reference: CLIP paper (https://arxiv.org/abs/2103.00020)
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        use_boxes: bool = False,
        use_hierarchies: bool = False,
        hier_sample_type: HierarchySampleType = HierarchySampleType.ALL,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        loss_fn="clip_loss"
    ):
        """
        Args:
            visual: ConvNet or ViT image encoder to compute image features.
            textual: Transformer-based encoder to compute text features.
            embed_dim: Size of the visual and textual embedding vectors for
                computing pairwise similarity matrix.
            pixel_mean: Normalize input images by this color mean. Default value
                is of ImageNet color, set to `(0, 0, 0)` for no normalization.
            pixel_std: Normalize input images by this color std. Default value
                is of ImageNet color, set to `(1, 1, 1)` for no normalization.
        """
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.embed_dim = embed_dim
        self.hier_sample_type = hier_sample_type

        # Linear layers to project image and text features such that they have
        # same size before computing dot-product similarity.
        self.visual_proj = nn.Linear(visual.width, embed_dim, bias=False)
        self.textual_proj = nn.Linear(textual.width, embed_dim, bias=False)

        # CLIP-style initialization of projection layers.
        nn.init.normal_(self.visual_proj.weight, std=visual.width**-0.5)
        nn.init.normal_(self.textual_proj.weight, std=textual.width**-0.5)

        # Initialize a learnable logit scale parameter.
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

        # Color mean/std to normalize image.
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1))

        # Get rank of current GPU process for gathering features.
        self._rank = dist.get_rank()
        self.loss_fn = losses.clip_loss

        if use_hierarchies:
            self.forward = self.forward_with_extra_samples
        elif use_boxes:
            self.forward = self.forward_with_boxes

        # otherwise, we use the default forward function
        # self.use_boxes = use_boxes
        # self.use_hierarchies = use_hierarchies
        print(f"INSIDE CLIP INIT, USE BOXES AND HIERARCHIES IS SET TO: {use_boxes} and {use_hierarchies} and hier_sample_type is set to {hier_sample_type}")

    @property
    def device(self) -> torch.device:
        return self.logit_scale.device

    def encode_image(self, images: torch.Tensor, project: bool):
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            project: Project features to a unit hypersphere through L2 normalization.

        Returns:
            Batch of image features of shape `(B, visual.width)`.
        """
        images = (images - self.pixel_mean) / self.pixel_std
        image_feats = self.visual(images)
        image_feats = self.visual_proj(image_feats)

        if project:
            image_feats = F.normalize(image_feats, dim=-1)

        return image_feats

    def encode_text(self, tokens: list[torch.Tensor], project: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Project features to a unit hypersphere through L2 normalization.
        """

        # Truncate tokens that are longer than context_length:
        for idx, inst_tokens in enumerate(tokens):
            if len(inst_tokens) > self.textual.context_length:
                eot_token = inst_tokens[-1]
                inst_tokens = inst_tokens[: self.textual.context_length]
                inst_tokens[-1] = eot_token
                tokens[idx] = inst_tokens

        # Pad all tokens on the right.
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        tokens = tokens.to(self.device)

        # shape: (batch_size, context_length, textual.width)
        text_feats = self.textual(tokens)

        # Get features for [EOS] position and apply projection. `[EOS]` token ID
        # is the largest number in the vocabulary of tokenizer.
        _eos_indices = tokens.argmax(dim=-1)
        batch_idxs = torch.arange(text_feats.shape[0])
        text_feats = text_feats[batch_idxs, _eos_indices]
        text_feats = self.textual_proj(text_feats)

        if project:
            text_feats = F.normalize(text_feats, dim=-1)

        return text_feats

    def forward(
        self, images: torch.Tensor, tokens: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        # Clamp temperature such that logits are not scaled more than 100x.
        # ln(100) = ~4.6052
        with torch.no_grad():
            self.logit_scale.clamp_(max=4.6052)
        _scale = self.logit_scale.exp()

        # Compute logits for image-text contrastive loss: cosine similarity.
        image_logits = _scale * image_feats @ all_text_feats.T
        text_logits = _scale * text_feats @ all_image_feats.T

        # Compute cross entropy loss: we compute log probabilities and take the
        # diagonal elements as targets: image[i] should match text[i] in batch.
        # Shift the targets according to rank of GPU process (we assume that all
        # GPU processes have the same local batch size).
        loss = self.loss_fn(
            image_logits, text_logits, self._rank, _scale
        )
        
        return loss
    
    def forward_with_boxes(
        self, images: torch.Tensor, box_images: torch.Tensor,
        tokens: list[torch.Tensor], box_tokens: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """
        # box_tokens shape: 192 x tensor
        # text_hierarchy_tokens shape: 192 x 4 x tensor

        # print("Forwarding with extra hierarchy samples! ")
        
        # Clamp temperature such that logits are not scaled more than 100x.
        # ln(100) = ~4.6052
        with torch.no_grad():
            self.logit_scale.clamp_(max=4.6052)
        _scale = self.logit_scale.exp()

        outputs = []
        # compute original I-T sample loss
        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)
        _rank = self._rank

        # Compute logits for image-text contrastive loss: cosine similarity.
        image_logits = _scale * image_feats @ all_text_feats.T
        text_logits = _scale * text_feats @ all_image_feats.T

        # Compute cross entropy loss: we compute log probabilities and take the
        # diagonal elements as targets: image[i] should match text[i] in batch.
        # Shift the targets according to rank of GPU process (we assume that all
        # GPU processes have the same local batch size).
        loss = self.loss_fn(
            image_logits, text_logits, _rank, _scale
        )
        outputs.append(loss)    
        
        extra_text_samples = [box_tokens]

        box_image_feats = self.encode_image(box_images, project=True)

        # compute loss for the extra samples
        for txt in extra_text_samples:
            # shape: (batch_size, embed_dim)
            txt_feats = self.encode_text(txt, project=True)

            # we consider these all to be 'box' samples:
            _rank = 0 # use this to get the local targets, i.e., don't use samples from other GPUs as negatives

            # Compute logits for image-text contrastive loss: cosine similarity.
            image_logits = _scale * box_image_feats @ txt_feats.T
            text_logits = _scale * txt_feats @ box_image_feats.T

            # Compute cross entropy loss: we compute log probabilities and take the
            # diagonal elements as targets: image[i] should match text[i] in batch.
            # Shift the targets according to rank of GPU process (we assume that all
            # GPU processes have the same local batch size).
            loss = self.loss_fn(
                image_logits, text_logits, _rank, _scale
            )
            outputs.append(loss)

        # loss = 0.5 * (outputs[0]["loss"] + outputs[1]["loss"])
        
        # compute average loss and contrastive loss across all samples
        loss = 0.0
        contrastive_loss = 0.0
        for output in outputs:
            loss += output["loss"]
            contrastive_loss += output["logging"]["contrastive_loss"]
        
        loss /= len(outputs)
        contrastive_loss /= len(outputs)

        return {
            "loss": loss,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "logit_scale": _scale,
            },
        }

    def forward_with_extra_samples(
        self, images: torch.Tensor, box_images: torch.Tensor,
        tokens: list[torch.Tensor], box_tokens: list[torch.Tensor],
        hierarchy_tokens: list[list[torch.Tensor]], pairwise_scores: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """
        # box_tokens shape: 192 x tensor
        # text_hierarchy_tokens shape: 192 x 4 x tensor

        # print("Forwarding with extra hierarchy samples! ")
        
        # Clamp temperature such that logits are not scaled more than 100x.
        # ln(100) = ~4.6052
        with torch.no_grad():
            self.logit_scale.clamp_(max=4.6052)
        _scale = self.logit_scale.exp()

        # first we need to transpose the 192 x 4 x tensor to be 4 x 192 x tensor
        transposed_hierarchy_tokens = [[row[i] for row in hierarchy_tokens] for i in range(len(hierarchy_tokens[0]))]
        # print(f"Outer T hier tokens list length: {len(transposed_hierarchy_tokens)}")
        # print(f"Inner T hier tokens list length: {len(transposed_hierarchy_tokens[0])}")
        # print(f"Inner Inner T hier tokens tensor shape: {transposed_hierarchy_tokens[0][0].shape}")
        
        outputs = []
        # compute original I-T sample loss
        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)
        _rank = self._rank

        # Compute logits for image-text contrastive loss: cosine similarity.
        image_logits = _scale * image_feats @ all_text_feats.T
        text_logits = _scale * text_feats @ all_image_feats.T

        # Compute cross entropy loss: we compute log probabilities and take the
        # diagonal elements as targets: image[i] should match text[i] in batch.
        # Shift the targets according to rank of GPU process (we assume that all
        # GPU processes have the same local batch size).
        loss = self.loss_fn(
            image_logits, text_logits, _rank, _scale
        )
        outputs.append(loss)

        if self.hier_sample_type == HierarchySampleType.ALL.value:
            # [192 x tensor] + 4 x 192 x tensor --> 5 x 192 x tensor 
            extra_text_samples = [box_tokens] + transposed_hierarchy_tokens
        elif self.hier_sample_type == HierarchySampleType.SINGLE_RANDOM.value:
            random_idx = torch.randint(0, len(transposed_hierarchy_tokens), (1,)).item()
            extra_text_samples = [box_tokens] + [transposed_hierarchy_tokens[random_idx]]
        else:
            raise ValueError(f"Invalid hier_sample_type: {self.hier_sample_type}. Must be either 'ALL' or 'SINGLE_RANDOM'.")
        # print(f"Outer extra sample  tokens list length: {len(extra_text_samples)}")
        # print(f"Inner extra sample list length: {len(extra_text_samples[0])}")
        # print(f"Inner extra sample tokens tensor shape: {extra_text_samples[0][0].shape}")
        # print(f"Total number of extra samples for a loss pass: {len(extra_text_samples)}")

        box_image_feats = self.encode_image(box_images, project=True)

        # compute loss for the extra samples
        for txt in extra_text_samples:
            # shape: (batch_size, embed_dim)
            txt_feats = self.encode_text(txt, project=True)

            # we consider these all to be 'box' samples:
            _rank = 0 # use this to get the local targets, i.e., don't use samples from other GPUs as negatives

            # Compute logits for image-text contrastive loss: cosine similarity.
            image_logits = _scale * box_image_feats @ txt_feats.T
            text_logits = _scale * txt_feats @ box_image_feats.T

            # Compute cross entropy loss: we compute log probabilities and take the
            # diagonal elements as targets: image[i] should match text[i] in batch.
            # Shift the targets according to rank of GPU process (we assume that all
            # GPU processes have the same local batch size).
            loss = self.loss_fn(
                image_logits, text_logits, _rank, _scale
            )
            outputs.append(loss)

        # loss = 0.5 * (outputs[0]["loss"] + outputs[1]["loss"])
        
        # compute average loss and contrastive loss across all samples
        loss = 0.0
        contrastive_loss = 0.0
        for output in outputs:
            loss += output["loss"]
            contrastive_loss += output["logging"]["contrastive_loss"]
        
        loss /= len(outputs)
        contrastive_loss /= len(outputs)

# DO NOT FORGET TO ALSO modify MERU INSTANTIATING CLIP because of new input params
# CHECK line-by-line whats the difference between such this models2 boxes implementation
# and the without box forward loop for clip. some of the torch operation might be making this much better.
# we will revert back to train.py from now on, but might use models2.py and losses.py cause its
# better structured, just make sure forward loop matches line-by-line with the one without boxes in models.py 

        return {
            "loss": loss,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "logit_scale": _scale,
            },
        }


class MERU(CLIPBaseline):
    """
    Implementation of MERU model that embeds images and text in a hyperbolic space.

    
    Reference: MERU paper (https://arxiv.org/abs/2304.09172)
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        use_boxes: bool = False,
        use_hierarchies: bool = False,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        loss_fn="meru_loss"
    ):
        """
        Un-documented args are same as `CLIPBaseline`.

        Args:
            curv_init: Positive scalar that denotes negative Hyperboloid curvature.
            learn_curv: Whether to learn the curvature parameter during training.
            entail_weight: Weight for the entailment loss component.
        """
        super().__init__(
            visual=visual,
            textual=textual,
            embed_dim=embed_dim,
            use_boxes=False,
            use_hierarchies=False,
            hier_sample_type=HierarchySampleType.ALL,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std)

        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        self.entail_weight = entail_weight

        # Learnable scalars to ensure that image/text features have an expected
        # unit norm before exponential map (at initialization).
        self.visual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.loss_fn = self.get_loss_fn(loss_fn)
        
        print(f"INSIDE MERU INIT, USE BOXES AND HIERARCHIES IS SET TO: {use_boxes} and {use_hierarchies}")

        if use_hierarchies:
            self.forward = self.forward_with_extra_samples
        elif use_boxes:
            self.forward = self.forward_with_boxes

    def encode_image(self, images: torch.Tensor, project: bool):
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            project: Lift features from the encoder onto the Hyperboloid.

        Returns:
            Batch of image features of shape `(B, visual.width)`.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        image_feats = super().encode_image(images, project=False)

        # These features are space components of embeddings in the tangent
        # space of the Hyperboloid origin (which is Euclidean). Apply projection.
        if project:
            image_feats = image_feats * self.visual_alpha.exp()
            image_feats = L.exp_map0(image_feats, self.curv.exp()) # all functions in L are decorated with autocast

        return image_feats

    def encode_text(self, tokens: list[torch.Tensor], project: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Lift features from the encoder onto the Hyperboloid.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        text_feats = super().encode_text(tokens, project=False)

        if project:
            text_feats = text_feats * self.textual_alpha.exp()
            text_feats = L.exp_map0(text_feats, self.curv.exp()) # all functions in L are decorated with autocast

        return text_feats

    def forward(
        self, images: torch.Tensor,
        tokens: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """
        with torch.no_grad():
            # Clamp scaling factors such that they do not up-scale the feature norms.
            # Once `exp(scale) = 1`, they can simply be removed during inference.
            self.visual_alpha.clamp_(max=0.0)
            self.textual_alpha.clamp_(max=0.0)
            self.logit_scale.clamp_(max=4.6052)
            self.curv.clamp_(**self._curv_minmax)
        _curv = self.curv.exp()
        _scale = self.logit_scale.exp()

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        # Compute all necessary loss components. All the loss functions in losses.py
        # are decorated with autocast to force a higher precision.
        loss = self.loss_fn(
            image_feats, text_feats, all_image_feats, all_text_feats,
            _curv, self._rank, _scale, entail_weight=self.entail_weight
        )
        
        return loss
    
    def get_loss_fn(self, loss_fn_name: str):
        """
        Returns the loss function based on the provided name.
        """
        if loss_fn_name == "meru_loss":
            return losses.meru_loss
        elif loss_fn_name == "meru_chord_loss":
            return losses.meru_chord_loss
        elif loss_fn_name == "meru_with_boxes_loss":
            return losses.meru_with_boxes_loss
        elif loss_fn_name == "accept_the_modality_gap_loss":
            return losses.accept_the_modality_gap_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_fn_name}.")

    def forward_with_boxes(
        self, images: torch.Tensor, box_images: torch.Tensor,
        tokens: list[torch.Tensor], box_tokens: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """
        # box_tokens shape: 192 x tensor
        
        # Clamp temperature such that logits are not scaled more than 100x.
        # ln(100) = ~4.6052
        with torch.no_grad():
            # Clamp scaling factors such that they do not up-scale the feature norms.
            # Once `exp(scale) = 1`, they can simply be removed during inference.
            self.visual_alpha.clamp_(max=0.0)
            self.textual_alpha.clamp_(max=0.0)
            self.logit_scale.clamp_(max=4.6052)
            self.curv.clamp_(**self._curv_minmax)
        _curv = self.curv.exp()
        _scale = self.logit_scale.exp()

        outputs = []
        # compute original I-T sample loss
        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)
        _rank = self._rank

        # Compute cross entropy loss: we compute log probabilities and take the
        # diagonal elements as targets: image[i] should match text[i] in batch.
        # Shift the targets according to rank of GPU process (we assume that all
        # GPU processes have the same local batch size).
        loss = self.loss_fn(
            image_feats, text_feats, all_image_feats, all_text_feats, 
            _curv, _rank, _scale, entail_weight=self.entail_weight
        )
        outputs.append(loss)    
        
        extra_text_samples = [box_tokens]

        box_image_feats = self.encode_image(box_images, project=True)

        # compute loss for the extra samples
        for txt in extra_text_samples:
            # shape: (batch_size, embed_dim)
            txt_feats = self.encode_text(txt, project=True)

            # we consider these all to be 'box' samples:
            _rank = 0 # use this to get the local targets, i.e., don't use samples from other GPUs as negatives

            # Compute cross entropy loss: we compute log probabilities and take the
            # diagonal elements as targets: image[i] should match text[i] in batch.
            # Shift the targets according to rank of GPU process (we assume that all
            # GPU processes have the same local batch size).
            loss = self.loss_fn(
                box_image_feats, txt_feats, image_feats, text_feats, 
                _curv, _rank, _scale, entail_weight=self.entail_weight
            )
            outputs.append(loss)
        
        # compute average loss and its components across all samples
        loss = 0.0
        contrastive_loss = 0.0
        entailment_loss = 0.0
        for output in outputs:
            loss += output["loss"]
            contrastive_loss += output["logging"]["contrastive_loss"]
            entailment_loss += output["logging"]["entailment_loss"]
        
        loss /= len(outputs)
        contrastive_loss /= len(outputs)
        entailment_loss /= len(outputs)

        return {
            "loss": loss,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "entailment_loss": entailment_loss,
                "logit_scale": _scale,
                "curv": _curv,
            },
        }
    
    # def forward_with_boxes(
    #     self, images: torch.Tensor, box_images: torch.Tensor,
    #     tokens: list[torch.Tensor], box_tokens: list[torch.Tensor]
    # ) -> dict[str, torch.Tensor]:
    #     """
    #     Args:
    #         images: Image batch in BCHW format, with pixel values in `[0, 1]`.
    #         tokens: List of tensors, each containing text tokens. Tensors may have
    #             variable length (they will be padded internally).
    #     """
    #     with torch.no_grad():
    #         # Clamp scaling factors such that they do not up-scale the feature norms.
    #         # Once `exp(scale) = 1`, they can simply be removed during inference.
    #         self.visual_alpha.clamp_(max=0.0)
    #         self.textual_alpha.clamp_(max=0.0)
    #         self.logit_scale.clamp_(max=4.6052)
    #         self.curv.clamp_(**self._curv_minmax)
    #     _curv = self.curv.exp()
    #     _scale = self.logit_scale.exp()

    #     inputs = [(images, tokens, False), (box_images, box_tokens, True)]
    #     outputs = []
    #     for (img, txt, is_box) in inputs:
    #         # shape: (batch_size, embed_dim)
    #         image_feats = self.encode_image(img, project=True)
    #         text_feats = self.encode_text(txt, project=True)

    #         if is_box:
    #             all_image_feats = image_feats
    #             all_text_feats = text_feats
    #             _rank = 0 # use this to get the local targets, i.e., don't use samples from other GPUs as negatives
    #         else:
    #             # Get features from all GPUs to increase negatives for contrastive loss.
    #             # These will be lists of tensors with length = world size.
    #             all_image_feats = dist.gather_across_processes(image_feats)
    #             all_text_feats = dist.gather_across_processes(text_feats)

    #             # shape: (batch_size * world_size, embed_dim)
    #             all_image_feats = torch.cat(all_image_feats, dim=0)
    #             all_text_feats = torch.cat(all_text_feats, dim=0)
    #             _rank = self._rank
    
    #         # Compute all necessary loss components. All the loss functions in losses.py
    #         # are decorated with autocast to force a higher precision.
    #         loss = self.loss_fn(
    #             image_feats, text_feats, all_image_feats, all_text_feats, 
    #             _curv, _rank, _scale, entail_weight=self.entail_weight
    #         )
    #         outputs.append(loss)

    #     loss = 0.5 * (outputs[0]["loss"] + outputs[1]["loss"])

    #     return {
    #         "loss": loss,
    #         "logging": {
    #             "contrastive_loss": 0.5 * (outputs[0]["logging"]["contrastive_loss"] + outputs[1]["logging"]["contrastive_loss"]),
    #             "entailment_loss": 0.5 * (outputs[0]["logging"]["entailment_loss"] + outputs[1]["logging"]["entailment_loss"]),
    #             "logit_scale": _scale,
    #             "curv": _curv,
    #         },
    #     }

    def forward_with_extra_samples(
        self, images: torch.Tensor, box_images: torch.Tensor,
        tokens: list[torch.Tensor], box_tokens: list[torch.Tensor],
        hierarchy_tokens: list[list[torch.Tensor]], pairwise_scores: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """
        # box_tokens shape: 192 x tensor
        
        # Clamp temperature such that logits are not scaled more than 100x.
        # ln(100) = ~4.6052
        with torch.no_grad():
            # Clamp scaling factors such that they do not up-scale the feature norms.
            # Once `exp(scale) = 1`, they can simply be removed during inference.
            self.visual_alpha.clamp_(max=0.0)
            self.textual_alpha.clamp_(max=0.0)
            self.logit_scale.clamp_(max=4.6052)
            self.curv.clamp_(**self._curv_minmax)
        _curv = self.curv.exp()
        _scale = self.logit_scale.exp()

        # first we need to transpose the 192 x 4 x tensor to be 4 x 192 x tensor
        transposed_hierarchy_tokens = [[row[i] for row in hierarchy_tokens] for i in range(len(hierarchy_tokens[0]))]
        # print(f"Outer T hier tokens list length: {len(transposed_hierarchy_tokens)}")
        # print(f"Inner T hier tokens list length: {len(transposed_hierarchy_tokens[0])}")
        # print(f"Inner Inner T hier tokens tensor shape: {transposed_hierarchy_tokens[0][0].shape}")
        
        outputs = []
        # compute original I-T sample loss
        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)
        _rank = self._rank

        # Compute cross entropy loss: we compute log probabilities and take the
        # diagonal elements as targets: image[i] should match text[i] in batch.
        # Shift the targets according to rank of GPU process (we assume that all
        # GPU processes have the same local batch size).
        loss = self.loss_fn(
            image_feats, text_feats, all_image_feats, all_text_feats, 
            _curv, _rank, _scale, entail_weight=self.entail_weight
        )
        outputs.append(loss)    
        
        # [192 x tensor] + 4 x 192 x tensor --> 5 x 192 x tensor 
        extra_text_samples = [box_tokens] + transposed_hierarchy_tokens
        # print(f"Outer extra sample  tokens list length: {len(extra_text_samples)}")
        # print(f"Inner extra sample list length: {len(extra_text_samples[0])}")
        # print(f"Inner extra sample tokens tensor shape: {extra_text_samples[0][0].shape}")
        # print(f"Total number of extra samples for a loss pass: {len(extra_text_samples)}")

        box_image_feats = self.encode_image(box_images, project=True)

        # compute loss for the extra samples
        for txt in extra_text_samples:
            # shape: (batch_size, embed_dim)
            txt_feats = self.encode_text(txt, project=True)

            # we consider these all to be 'box' samples:
            _rank = 0 # use this to get the local targets, i.e., don't use samples from other GPUs as negatives

            # Compute cross entropy loss: we compute log probabilities and take the
            # diagonal elements as targets: image[i] should match text[i] in batch.
            # Shift the targets according to rank of GPU process (we assume that all
            # GPU processes have the same local batch size).
            loss = self.loss_fn(
                box_image_feats, txt_feats, image_feats, text_feats, 
                _curv, _rank, _scale, entail_weight=self.entail_weight
            )
            outputs.append(loss)
        
        # compute average loss and its components across all samples
        loss = 0.0
        contrastive_loss = 0.0
        entailment_loss = 0.0
        for output in outputs:
            loss += output["loss"]
            contrastive_loss += output["logging"]["contrastive_loss"]
            entailment_loss += output["logging"]["entailment_loss"]
        
        loss /= len(outputs)
        contrastive_loss /= len(outputs)
        entailment_loss /= len(outputs)

        return {
            "loss": loss,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "entailment_loss": entailment_loss,
                "logit_scale": _scale,
                "curv": _curv,
            },
        }
    

class HyCoCLIP(MERU):
    """
    Our HyCoCLIP model, that modifies MERU and CLIP to embed images, texts and their localized box 
    information hierarchically in a hyperbolic space.
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        use_boxes: bool = True,
        use_hierarchies: bool = False,
        hier_sample_type: HierarchySampleType = HierarchySampleType.ALL,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        loss_fn="hycoclip_loss"
    ):
        """
        Un-documented args are same as `MERU`.

        Args:
            use_boxes: Whether to use box images and texts for training.
        """
        super().__init__(visual=visual,
                        textual=textual,
                        embed_dim=embed_dim,
                        curv_init=curv_init,
                        learn_curv=learn_curv,
                        entail_weight=entail_weight,
                        use_boxes=False,
                        use_hierarchies=False,
                        pixel_mean=pixel_mean,
                        pixel_std=pixel_std,
                        loss_fn=loss_fn)
    
        assert use_boxes, "HyCoCLIP requires box images and texts to function."

        self.hier_sample_type = hier_sample_type

        print(f"Initialized HyCoCLIP with loss function: {loss_fn}")
        print(f"Use boxes: {use_boxes}, Use hierarchies: {use_hierarchies}")

        if use_hierarchies:
            print(f"Hierarchy sample type set to: {hier_sample_type}")
            self.forward = self.forward_with_extra_samples

        self.loss_fn = self.get_loss_fn(loss_fn)


    def forward(
        self, images: torch.Tensor, box_images: torch.Tensor,
        tokens: list[torch.Tensor], box_tokens: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """
        with torch.no_grad():
            # Clamp scaling factors such that they do not up-scale the feature norms.
            # Once `exp(scale) = 1`, they can simply be removed during inference.
            self.visual_alpha.clamp_(max=0.0)
            self.textual_alpha.clamp_(max=0.0)
            # Clamp the curvature parameter to prevent instability.
            self.curv.clamp_(**self._curv_minmax)
            # Clamp temperature such that logits are not scaled more than 100x.
            # ln(100) = ~4.6052
            self.logit_scale.clamp_(max=4.6052)
        _curv = self.curv.exp()
        _scale = self.logit_scale.exp()

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        box_image_feats = self.encode_image(box_images, project=True)
        box_text_feats = self.encode_text(box_tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        # Compute all necessary loss components. All the loss functions in losses.py
        # are decorated with autocast to force a higher precision.
        loss = self.loss_fn(
            image_feats, text_feats, box_image_feats, box_text_feats,
            all_image_feats, all_text_feats, _curv, self._rank, _scale,
            entail_weight=self.entail_weight
        )
        
        return loss
    
    def forward_with_extra_samples(
        self, images: torch.Tensor, box_images: torch.Tensor,
        tokens: list[torch.Tensor], box_tokens: list[torch.Tensor],
        hierarchy_tokens: list[list[torch.Tensor]], pairwise_scores: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """
        with torch.no_grad():
            # Clamp scaling factors such that they do not up-scale the feature norms.
            # Once `exp(scale) = 1`, they can simply be removed during inference.
            self.visual_alpha.clamp_(max=0.0)
            self.textual_alpha.clamp_(max=0.0)
            # Clamp the curvature parameter to prevent instability.
            self.curv.clamp_(**self._curv_minmax)
            # Clamp temperature such that logits are not scaled more than 100x.
            # ln(100) = ~4.6052
            self.logit_scale.clamp_(max=4.6052)
        _curv = self.curv.exp()
        _scale = self.logit_scale.exp()

        # first we need to transpose the 192 x 4 x tensor to be 4 x 192 x tensor
        transposed_hierarchy_tokens = [[row[i] for row in hierarchy_tokens] for i in range(len(hierarchy_tokens[0]))]
        # print(f"Outer T hier tokens list length: {len(transposed_hierarchy_tokens)}")
        # print(f"Inner T hier tokens list length: {len(transposed_hierarchy_tokens[0])}")
        # print(f"Inner Inner T hier tokens tensor shape: {transposed_hierarchy_tokens[0][0].shape}")

        # compute hierarchy features. shape: (4, batch_size, embed_dim)
        hierarchy_feats_list = [self.encode_text(txt, project=True) for txt in transposed_hierarchy_tokens]

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        box_image_feats = self.encode_image(box_images, project=True)
        box_text_feats = self.encode_text(box_tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        # Compute all necessary loss components. All the loss functions in losses.py
        # are decorated with autocast to force a higher precision.
        loss = self.loss_fn(
            image_feats, text_feats, box_image_feats, box_text_feats, hierarchy_feats_list, self.hier_sample_type,
            all_image_feats, all_text_feats, _curv, self._rank, _scale,
            entail_weight=self.entail_weight,
        )
        
        return loss

    # def forward_with_extra_samples(
    #     self, images: torch.Tensor, box_images: torch.Tensor,
    #     tokens: list[torch.Tensor], box_tokens: list[torch.Tensor],
    #     hierarchy_tokens: list[list[torch.Tensor]], pairwise_scores: torch.Tensor
    # ) -> dict[str, torch.Tensor]:
    #     """
    #     Args:
    #         images: Image batch in BCHW format, with pixel values in `[0, 1]`.
    #         tokens: List of tensors, each containing text tokens. Tensors may have
    #             variable length (they will be padded internally).
    #     """
    #     print("Forwarding with extra hierarchy samples! ")

    #     with torch.no_grad():
    #         # Clamp scaling factors such that they do not up-scale the feature norms.
    #         # Once `exp(scale) = 1`, they can simply be removed during inference.
    #         self.visual_alpha.clamp_(max=0.0)
    #         self.textual_alpha.clamp_(max=0.0)
    #         # Clamp the curvature parameter to prevent instability.
    #         self.curv.clamp_(**self._curv_minmax)
    #         # Clamp temperature such that logits are not scaled more than 100x.
    #         # ln(100) = ~4.6052
    #         self.logit_scale.clamp_(max=4.6052)
    #     _curv = self.curv.exp()
    #     _scale = self.logit_scale.exp()

    #     # first we need to transpose the 192 x 4 x tensor to be 4 x 192 x tensor
    #     transposed_hierarchy_tokens = [[row[i] for row in hierarchy_tokens] for i in range(len(hierarchy_tokens[0]))]
    #     # print(f"Outer T hier tokens list length: {len(transposed_hierarchy_tokens)}")
    #     # print(f"Inner T hier tokens list length: {len(transposed_hierarchy_tokens[0])}")
    #     # print(f"Inner Inner T hier tokens tensor shape: {transposed_hierarchy_tokens[0][0].shape}")
        
    #     outputs = []

    #     # shape: (batch_size, embed_dim)
    #     image_feats = self.encode_image(images, project=True)
    #     text_feats = self.encode_text(tokens, project=True)

    #     box_image_feats = self.encode_image(box_images, project=True)
    #     box_text_feats = self.encode_text(box_tokens, project=True)

    #     # Get features from all GPUs to increase negatives for contrastive loss.
    #     # These will be lists of tensors with length = world size.
    #     all_image_feats = dist.gather_across_processes(image_feats)
    #     all_text_feats = dist.gather_across_processes(text_feats)

    #     # shape: (batch_size * world_size, embed_dim)
    #     all_image_feats = torch.cat(all_image_feats, dim=0)
    #     all_text_feats = torch.cat(all_text_feats, dim=0)

    #     print("Forwarding with extra hierarchy samples! 2")

    #     # Compute all necessary loss components. All the loss functions in losses.py
    #     # are decorated with autocast to force a higher precision.
    #     loss = self.loss_fn(
    #         image_feats, text_feats, box_image_feats, box_text_feats,
    #         all_image_feats, all_text_feats, _curv, self._rank, _scale,
    #         entail_weight=self.entail_weight
    #     )
    #     outputs.append(loss)    
        
    #     print("Forwarding with extra hierarchy samples 3! ")

    #     # [192 x tensor] + 4 x 192 x tensor --> 5 x 192 x tensor 
    #     if self.hier_sample_type == HierarchySampleType.ALL.value:
    #         # [192 x tensor] + 4 x 192 x tensor --> 5 x 192 x tensor 
    #         extra_text_samples = transposed_hierarchy_tokens #[box_tokens] + transposed_hierarchy_tokens
    #     elif self.hier_sample_type == HierarchySampleType.SINGLE_RANDOM.value:
    #         random_idx = torch.randint(0, len(transposed_hierarchy_tokens), (1,)).item()
    #         extra_text_samples = transposed_hierarchy_tokens[random_idx] #[box_tokens] + [transposed_hierarchy_tokens[random_idx]]
    #     else:
    #         raise ValueError(f"Invalid hier_sample_type: {self.hier_sample_type}. Must be either 'ALL' or 'SINGLE_RANDOM'.")
    #     # print(f"Outer extra sample  tokens list length: {len(extra_text_samples)}")
    #     # print(f"Inner extra sample list length: {len(extra_text_samples[0])}")
    #     # print(f"Inner extra sample tokens tensor shape: {extra_text_samples[0][0].shape}")
    #     # print(f"Total number of extra samples for a loss pass: {len(extra_text_samples)}")

    #     box_image_feats = self.encode_image(box_images, project=True)

    #     # compute loss for the extra samples
    #     for txt in extra_text_samples:
    #         # shape: (batch_size, embed_dim)
    #         extra_txt_feats = self.encode_text(txt, project=True)

    #         # we consider these all to be 'box' samples:
    #         _rank = 0 # use this to get the local targets, i.e., don't use samples from other GPUs as negatives

    #         # Compute cross entropy loss: we compute log probabilities and take the
    #         # diagonal elements as targets: image[i] should match text[i] in batch.
    #         # Shift the targets according to rank of GPU process (we assume that all
    #         # GPU processes have the same local batch size).
    #         loss = self.loss_fn(
    #             image_feats, text_feats, box_image_feats, extra_txt_feats,
    #             all_image_feats, all_text_feats, _curv, _rank, _scale,
    #             entail_weight=self.entail_weight
    #         )
    #         outputs.append(loss)
        
    #     print("Forwarding with extra hierarchy samples! 4")

    #     # compute average loss and logging stats across all samples
    #     n = len(outputs)
    #     # loss = 0.0
    #     # contrastive_loss = 0.0
    #     # entailment_loss = 0.0
    #     # text_image_entailment_loss = 0.0
    #     # box_text_image_entailment_loss = 0.0
    #     # cross_image_entailment_loss = 0.0
    #     # cross_text_entailment_loss = 0.0
        
    #     # for output in outputs:
    #     #     loss += output["loss"]
    #     #     contrastive_loss += output["logging"]["contrastive_loss"]
    #     #     entailment_loss += output["logging"]["entailment_loss"]
    #     #     text_image_entailment_loss += output["logging"]["text_image_entailment_loss"]
    #     #     box_text_image_entailment_loss += output["logging"]["box_text_image_entailment_loss"]
    #     #     cross_image_entailment_loss += output["logging"]["cross_image_entailment_loss"]
    #     #     cross_text_entailment_loss += output["logging"]["cross_text_entailment_loss"]
        
    #     # loss /= n
    #     # contrastive_loss /= n
    #     # entailment_loss /= n
    #     # text_image_entailment_loss /= n
    #     # box_text_image_entailment_loss /= n
    #     # cross_image_entailment_loss /= n
    #     # cross_text_entailment_loss /= n

    #     loss = sum(o["loss"] for o in outputs) / n
    #     # average each logged component in one comprehension
    #     avg_logs = {
    #         key: sum(o["logging"][key] for o in outputs) / n
    #         for key in (
    #             "contrastive_loss",
    #             "entailment_loss",
    #             "text_image_entailment_loss",
    #             "box_text_image_entailment_loss",
    #             "cross_image_entailment_loss",
    #             "cross_text_entailment_loss",
    #         )
    #     }
    #     contrastive_loss = avg_logs["contrastive_loss"]
    #     entailment_loss = avg_logs["entailment_loss"]
    #     text_image_entailment_loss = avg_logs["text_image_entailment_loss"]
    #     box_text_image_entailment_loss = avg_logs["box_text_image_entailment_loss"]
    #     cross_image_entailment_loss = avg_logs["cross_image_entailment_loss"]
    #     cross_text_entailment_loss = avg_logs["cross_text_entailment_loss"]

    #     print("Forwarding with extra hierarchy samples! 5")

    #     return {
    #         "loss": loss,
    #         "logging": {
    #             "contrastive_loss": contrastive_loss,
    #             "text_image_entailment_loss": text_image_entailment_loss,
    #             "box_text_image_entailment_loss": box_text_image_entailment_loss,
    #             "cross_image_entailment_loss": cross_image_entailment_loss,
    #             "cross_text_entailment_loss": cross_text_entailment_loss,
    #             "entailment_loss": entailment_loss,
    #             "logit_scale": _scale,
    #             "curv": _curv,
    #         },
    #     }
        

    def get_loss_fn(self, loss_fn_name: str):
        """
        Returns the loss function based on the provided name.
        """
        if loss_fn_name == "hycoclip_loss":
            return losses.hycoclip_loss
        elif loss_fn_name == "chordclip_loss":
            return losses.chordclip_loss
        elif loss_fn_name == "hycoclip_deep_loss":
            return losses.hycoclip_deep_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_fn_name}.")
        
    def reinit_projection(self):
        """
        Re-initialize the projection layers and logit scale parameter.
        """
        nn.init.normal_(self.visual_proj.weight, std=self.visual.width**-0.5)
        nn.init.normal_(self.textual_proj.weight, std=self.textual.width**-0.5)
        self.logit_scale.data = torch.tensor(1 / 0.07).log()
        self.curv.data = torch.tensor(1.0).log()
        self.visual_alpha.data = torch.tensor(self.embed_dim**-0.5).log()
        self.textual_alpha.data = torch.tensor(self.embed_dim**-0.5).log()
        print("Re-initialized projection layers and logit scale parameter.")



class HyCoCLIP_Re_Weight(MERU):
    """
    Our HyCoCLIP_Re_Weight model, that modifies MERU and CLIP to embed images, texts and their localized box 
    information hierarchically in a hyperbolic space.
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
        loss_fn="hyco_reweight_loss"
    ):
        """
        Un-documented args are same as `MERU`.

        Args:
            use_boxes: Whether to use box images and texts for training.
        """
        super().__init__(visual=visual,
                         textual=textual,
                         embed_dim=embed_dim,
                         curv_init=curv_init,
                         learn_curv=learn_curv,
                         entail_weight=entail_weight,
                         use_boxes=False,
                         use_hierarchies=False,
                         pixel_mean=pixel_mean,
                         pixel_std=pixel_std,
                         loss_fn=loss_fn)
        
        assert use_hierarchies, "HyCoCLIP ReWeight requires textual hierarchies to function."
        self.contrast_weight = contrast_weight
        self.loss_fn = self.get_loss_fn(loss_fn)

        print(f"INSIDE RE WEIGHT INIT, USE HIERARCHIES IS SET TO: {use_hierarchies}")

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
        """
        with torch.no_grad():
            # Clamp scaling factors such that they do not up-scale the feature norms.
            # Once `exp(scale) = 1`, they can simply be removed during inference.
            self.visual_alpha.clamp_(max=0.0)
            self.textual_alpha.clamp_(max=0.0)
            # Clamp the curvature parameter to prevent instability.
            self.curv.clamp_(**self._curv_minmax)
            # Clamp temperature such that logits are not scaled more than 100x.
            # ln(100) = ~4.6052
            self.logit_scale.clamp_(max=4.6052)
        _curv = self.curv.exp()
        _scale = self.logit_scale.exp()

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        box_image_feats = self.encode_image(box_images, project=True)
        box_text_feats = self.encode_text(box_tokens, project=True)

        # first we need to transpose the 192 x 4 x tensor to be 4 x 192 x tensor
        transposed_hier_tokens = [[row[i] for row in hierarchy_tokens] for i in range(len(hierarchy_tokens[0]))]
        # compute hierarchy features
        hierarchy_feats = [self.encode_text(transposed_hier_tokens[i], project=True) for i in range(len(transposed_hier_tokens))]
        # old, probably incorrect:
        # hierarchy_feats = [self.encode_text([hier[i] for hier in hierarchy_tokens], project=True) for i in range(4)]

        # transpose the 192 x 4 pairwise scores tensor to be 4 x 192 tensor
        pairwise_scores = pairwise_scores.T
        # print(f"PAIRWISE SCORE SHAPE: {pairwise_scores.shape}")
        # print(f"PAIRWISE SCORE SHAPE: {pairwise_scores[0].shape}")

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        # Compute all necessary loss components. All the loss functions in losses.py
        # are decorated with autocast to force a higher precision.
        loss = self.loss_fn(
            image_feats, text_feats, box_image_feats, box_text_feats,
            all_image_feats, all_text_feats, hierarchy_feats, pairwise_scores, _curv, self._rank, _scale,
            entail_weight=self.entail_weight, contrast_weight=self.contrast_weight
        )
        
        return loss
    
    def get_loss_fn(self, loss_fn_name: str):
        """
        Returns the loss function based on the provided name.
        """
        if loss_fn_name == "hyco_reweight_loss":
            return losses.hyco_reweight_loss 
        elif loss_fn_name == "hyco_reweight_slightly_smaller_K_loss":
            return losses.hyco_reweight_loss_slightly_smaller_K # FOR TESTING K 
        elif loss_fn_name == "hyco_reweight_very_small_K_loss":
            return losses.hyco_reweight_loss_very_small_K # FOR TESTING K 
        else:
            raise ValueError(f"Unknown loss function: {loss_fn_name}.")
        
    def reinit_projection(self):
        """
        Re-initialize the projection layers and logit scale parameter.
        """
        nn.init.normal_(self.visual_proj.weight, std=self.visual.width**-0.5)
        nn.init.normal_(self.textual_proj.weight, std=self.textual.width**-0.5)
        self.logit_scale.data = torch.tensor(1 / 0.07).log()
        self.curv.data = torch.tensor(1.0).log()
        self.visual_alpha.data = torch.tensor(self.embed_dim**-0.5).log()
        self.textual_alpha.data = torch.tensor(self.embed_dim**-0.5).log()
        print("Re-initialized projection layers and logit scale parameter.")
