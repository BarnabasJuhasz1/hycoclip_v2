"""
MERU training on the same mixed GRIT (child-only) + pixmo-cap (flat) dataset
as train_clip_grit_pixmocap_vit_b.py — see that file for dataset details.
"""

from hycoclip.config import LazyCall as L
from hycoclip.encoders.image_encoders import build_timm_vit
from hycoclip.encoders.text_encoders import TransformerTextEncoder
from hycoclip.models2 import MERU

from .train_clip_grit_pixmocap_vit_b import dataset, optim, train


model = L(MERU)(
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
    use_boxes=False,
    use_hierarchies=False,
    loss_fn="meru_loss",
)
