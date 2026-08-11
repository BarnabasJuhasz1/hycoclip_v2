"""
HyCoCLIP Deep (proposed hierarchies) training with polynomial repulsion loss.
Uses max(0, r0 - r)^2 repulsion on top of the Deep-GRIT hierarchy loss.
"""

from torchvision import transforms as T

from hycoclip.config import LazyCall as L
from hycoclip.data.webdataset_mapper import ExtendedGroundedDatasetTarMapper, ImageTextWebDataset

from .train_hycoclip_vit_l import model, optim, train


# The base config's mapper only serves boxes; hierarchies need the extended one.
dataset = L(ImageTextWebDataset)(
    tarfiles=["datasets/train/GRIT/tar_fitted/*.tar"],
    mapper=L(ExtendedGroundedDatasetTarMapper)(
        image_transform=[
            L(T.RandomResizedCrop)(
                size=224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC
            ),
            L(T.ToTensor)(),
        ],
        # change to enable/disable the use of proposed hierarchies when available
        use_proposed_hierachies=True,
    ),
    buffer_size=4000,
    seed="${..train.seed}",
)


model.visual.arch = "vit_base_patch16_224"


# USING boxes by default when using hierarchies
model.use_boxes = True

# USING Deep-GRIT hierarchies
model.use_hierarchies = True
# using hierarchies with the added repulsion term by setting the loss
model.loss_fn = "hycoclip_deep_loss_repulsion_poly"

# using all hierarchy sample
model.hier_sample_type = "SINGLE_RANDOM" # one of "SINGLE_RANDOM" or "ALL"

model.repulsion_weight = 0.1  # Weight for the repulsion loss component
model.repulsion_r0 = 0.5      # Target minimum norm: only penalize when r < 0.5
