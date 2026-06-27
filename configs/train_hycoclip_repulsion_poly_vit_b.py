"""
HyCoCLIP training with polynomial repulsion loss.
Uses max(0, r0 - r)^2 repulsion instead of exponential e^(-r).
Better control over target embedding separation distance.
"""

from torch.optim import AdamW
from torchvision import transforms as T

from hycoclip.config import LazyCall as L
from hycoclip.data.webdataset_mapper import GroundedDatasetTarMapper, ImageTextWebDataset
from hycoclip.encoders.image_encoders import build_timm_vit
from hycoclip.encoders.text_encoders import TransformerTextEncoder
from hycoclip.models import HyCoCLIP_Repulsion_Poly
from hycoclip.optim import LinearWarmupCosineDecayLR, set_weight_decay_per_param


dataset = L(ImageTextWebDataset)(
    tarfiles=["datasets/train/GRIT/tar_fitted/*.tar"],
    mapper=L(GroundedDatasetTarMapper)(
        image_transform=[
            L(T.RandomResizedCrop)(
                size=224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC
            ),
            L(T.ToTensor)(),
        ],
    ),
    buffer_size=4000,
    initial_buffer_size=500,
    seed="${..train.seed}",
)


model = L(HyCoCLIP_Repulsion_Poly)(
    visual=L(build_timm_vit)(
        arch="vit_base_patch16_224",
        global_pool="token",
        use_sincos2d_pos=True,
        grad_checkpointing=True,
    ),
    textual=L(TransformerTextEncoder)(
        arch="L12_W512", vocab_size=49408, context_length=77
    ),
    embed_dim=512,
    curv_init=1.0,
    learn_curv=True,
    entail_weight=0.2,
    repulsion_weight=0.1,  # Weight for the repulsion loss component
    repulsion_r0=0.5,      # Target minimum norm: only penalize when r < 0.5
    use_boxes=True,
    use_hierarchies=False,  # Not using Deep-GRIT hierarchies
    loss_fn="hycoclip_loss_repulsion_poly",  # Use polynomial repulsion loss
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
    num_iterations=90000,
    cudnn_benchmark=True,
    cudnn_deterministic=False,
    num_workers=4,
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
    ),
    ddp_fp16_compression=True,
)
