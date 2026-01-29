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

from torchvision import transforms as T

from hycoclip.config import LazyCall as L
from hycoclip.data.webdataset_mapper import GroundedDatasetTarMapperV2, ImageTextWebDataset
from hycoclip.encoders.image_encoders import build_timm_vit
from hycoclip.encoders.text_encoders import TransformerTextEncoder
from hycoclip.new_models.re_weight_DinContrastive import HyCoCLIP_Re_Weight_DinContrastive

from .train_hycoclip_vit_l import optim, train

dataset = L(ImageTextWebDataset)(
    tarfiles=["datasets/train/GRIT/tar_fitted/*.tar"],
    mapper=L(GroundedDatasetTarMapperV2)(
        hierfiles=["datasets/train/hier_dataset/json_pairwise/*.json"],
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


model = L(HyCoCLIP_Re_Weight_DinContrastive)(
    visual=L(build_timm_vit)(
        arch="vit_base_patch16_224",
        global_pool="token",
        use_sincos2d_pos=True,
    ),
    textual=L(TransformerTextEncoder)(
        arch="L12_W512", vocab_size=49408, context_length=77
    ),
    embed_dim=512,
    curv_init=1.0,
    learn_curv=True,
    entail_weight=0.2,
    use_boxes=True,
    use_hierarchies=True
)
