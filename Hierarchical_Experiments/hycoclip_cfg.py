#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

# Modified from github.com/facebookresearch/meru

"""
Each config file should have four dicts or OmegaConf objects:
`dataset`, `model`, `optim`, and `train`.

User can compose config files by importing these objects and overriding specific
parameters. See examples in other training configs.

Reference: https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html
"""

from torch.optim import AdamW
from torchvision import transforms as T

from hycoclip.config import LazyCall as L
from hycoclip.data.webdataset_mapper import GroundedDatasetTarMapper, ImageTextWebDataset, GroundedDatasetTarMapperV2
from hycoclip.encoders.image_encoders import build_timm_vit
from hycoclip.encoders.text_encoders import TransformerTextEncoder
from hycoclip.models import HyCoCLIP_V2
from hycoclip.models import HyCoCLIP_V2_EASY
from hycoclip.optim import LinearWarmupCosineDecayLR, set_weight_decay_per_param


dataset = L(ImageTextWebDataset)(
    tarfiles=["datasets/train/GRIT/tar_fitted/*.tar"],
    mapper=L(GroundedDatasetTarMapperV2)(
        hierfiles=["datasets/train/hier_dataset/json_intermediary/*.json"],
        image_transform=[
            L(T.RandomResizedCrop)(
                size=224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC
            ),
            L(T.ToTensor)(),
        ],  
    ),
    buffer_size=4000,
    seed="${..train.seed}",
)


model = L(HyCoCLIP_V2)(
    visual=L(build_timm_vit)(
        arch="vit_large_patch16_224",
        global_pool="token",
        use_sincos2d_pos=True,
    ),
    textual=L(TransformerTextEncoder)(
        arch="L12_W512", vocab_size=49408, context_length=77 # originally context_length=77
    ),
    embed_dim=512,
    curv_init=1.0,
    learn_curv=True,
    entail_weight=0.2, #0.2,
    use_boxes=True,
    use_hierarchies=True,
    cont_weights=[1.25, 1, 0.75, 0.5],
    #ent_cone_scale=[1.6, 1.5, 1.4, 1.3],
)


# AdamW with no weight decay for norm, bias, and other learnable scalars.
optim = dict(
    optimizer=L(AdamW)(
        params=L(set_weight_decay_per_param)(
            weight_decay="${..weight_decay}",
            gain_bias_decay=0.0,
            exclude_params=[
                "logit_scale", "visual_alpha", "textual_alpha", "curv"
            ],
        ),
        lr=5e-4,
        #lr=2.5e-5,
        # lr=1e-3,
        betas=(0.9, 0.98),
        weight_decay=0.2,
    ),
    lr_scheduler=L(LinearWarmupCosineDecayLR)(
        total_steps="${...train.num_iterations}", warmup_steps=4000
    ),
)


# Other parameters useful for training script.
train = dict(
    seed=0,
    amp=True,
    total_batch_size=768,
    num_iterations=180000, #500000,
    cudnn_benchmark=True,
    cudnn_deterministic=False,
    num_workers=4,
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False, static_graph=True
    ),
    ddp_fp16_compression=True,
)

model.visual.arch = "vit_base_patch16_224"