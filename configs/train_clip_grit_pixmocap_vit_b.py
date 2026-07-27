

from torch.optim import AdamW
from torchvision import transforms as T

from hycoclip.config import LazyCall as L
from hycoclip.data.webdataset_mapper import (
    ChildOnlyTarMapper,
    FlatImageTextWebDataset,
    ImageTextTarMapper,
    ImageTextWebDataset,
    MixedIterableDataset,
    check_child_keys_raw,
)
from hycoclip.encoders.image_encoders import build_timm_vit
from hycoclip.encoders.text_encoders import TransformerTextEncoder
from hycoclip.models2 import CLIPBaseline
from hycoclip.optim import LinearWarmupCosineDecayLR, set_weight_decay_per_param


# Fixed (not tied to train.seed via interpolation, unlike the top-level mix
# seed below) to avoid ambiguous relative-interpolation depth through the
# `datasets` list.
_SUBDATASET_SEED = 0

_image_transform = [
    L(T.RandomResizedCrop)(
        size=224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC
    ),
    L(T.ToTensor)(),
]

dataset = L(MixedIterableDataset)(
    datasets=[
        L(ImageTextWebDataset)(
            tarfiles=["datasets/train/GRIT/tar_fitted/*.tar"],
            mapper=L(ChildOnlyTarMapper)(image_transform=_image_transform),
            sample_filter=check_child_keys_raw,
            buffer_size=4000,
            initial_buffer_size=500,
            seed=_SUBDATASET_SEED,
        ),
        L(FlatImageTextWebDataset)(
            tarfiles=["datasets/train/PixmoCap/img2dataset_raw/*.tar"],
            mapper=L(ImageTextTarMapper)(image_transform=_image_transform),
            buffer_size=2000,
            initial_buffer_size=500,
            seed=_SUBDATASET_SEED,
        ),
    ],
    weights=[0.0, 1.0], #only pixmo
    seed="${..train.seed}",
)


model = L(CLIPBaseline)(
    visual=L(build_timm_vit)(
        arch="vit_base_patch16_224",
        global_pool="token",
        use_sincos2d_pos=True,
    ),
    textual=L(TransformerTextEncoder)(
        arch="L12_W512", vocab_size=49408, context_length=77
    ),
    embed_dim=512,
    use_boxes=False,  # neither GRIT (child-only) nor pixmo-cap have boxes here
    use_hierarchies=False,
    hier_sample_type="SINGLE_RANDOM",
    loss_fn="clip_loss",
)


# AdamW with no weight decay for norm, bias, and other learnable scalars.
optim = dict(
    optimizer=L(AdamW)(
        params=L(set_weight_decay_per_param)(
            weight_decay="${..weight_decay}",
            gain_bias_decay=0.0,
            exclude_params=["logit_scale"],
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
    ddp=dict(broadcast_buffers=False, static_graph=True),
    ddp_fp16_compression=True,
)
