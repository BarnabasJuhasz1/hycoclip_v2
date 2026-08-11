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
        use_proposed_hierachies=False, # NOT using proposed hierarchies by default
    ),
    buffer_size=4000,
    seed="${..train.seed}",
)


model.visual.arch = "vit_base_patch16_224"


# USING boxes by default when using hierarchies
model.use_boxes = True

# USING Deep-GRIT hierarchies
model.use_hierarchies = True
# using hierarchies by setting the loss
model.loss_fn = "hycoclip_deep_loss"

# using all hierarchy sample
model.hier_sample_type = "SINGLE_RANDOM" # one of "SINGLE_RANDOM" or "ALL"
