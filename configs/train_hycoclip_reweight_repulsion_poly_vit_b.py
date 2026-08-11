"""
HyCoCLIP ReWeight training with polynomial repulsion loss.
Uses max(0, r0 - r)^2 repulsion on top of the score-modulated hierarchy loss.
"""

from torchvision import transforms as T

from hycoclip.config import LazyCall as L
from hycoclip.data.webdataset_mapper import ExtendedGroundedDatasetTarMapper, ImageTextWebDataset
from hycoclip.encoders.image_encoders import build_timm_vit
from hycoclip.encoders.text_encoders import TransformerTextEncoder
from hycoclip.models2 import HyCoCLIP_Re_Weight

from .train_hycoclip_vit_l import optim, train

dataset = L(ImageTextWebDataset)(
    tarfiles=["datasets/train/GRIT/tar_fitted/*.tar"],
    mapper=L(ExtendedGroundedDatasetTarMapper)(
        image_transform=[
            L(T.RandomResizedCrop)(
                size=224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC
            ),
            L(T.ToTensor)(),
        ],
        use_proposed_hierachies=False, # NOT using proposed hierarchies by default
    ),
    buffer_size=4000,
    seed="${..train.seed}",
)

model = L(HyCoCLIP_Re_Weight)(
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
    entail_weight=0.1,
    contrast_weight=0.25,
    # use_boxes=True, # not necessary because we always use boxes
    use_hierarchies=True,
    loss_fn="hyco_reweight_loss_repulsion_poly",
    repulsion_weight=0.1,  # Weight for the repulsion loss component
    repulsion_r0=0.5,      # Target minimum norm: only penalize when r < 0.5
)
