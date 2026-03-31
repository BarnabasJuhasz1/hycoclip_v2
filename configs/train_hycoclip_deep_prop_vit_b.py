from .train_hycoclip_vit_l import dataset, model, optim, train


model.visual.arch = "vit_base_patch16_224"


# USING boxes by default when using hierarchies
model.use_boxes = True

# USING Deep-GRIT hierarchies
model.use_hierarchies = True
# using hierarchies by setting the loss
model.loss_fn = "hycoclip_deep_loss"

# using all hierarchy sample
model.hier_sample_type = "SINGLE_RANDOM" # one of "SINGLE_RANDOM" or "ALL"

# change to enable/disable the use of proposed hierarchies when available
dataset.mapper.use_proposed_hierachies = True
