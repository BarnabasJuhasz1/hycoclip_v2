#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

from .train_clip_vit_l import dataset, model, optim, train



model.visual.arch = "vit_base_patch16_224"

# USING boxes by default when using hierarchies
model.use_boxes = True

# USING Deep-GRIT hierarchies
# dataset.mapper.use_hierarchies = True
model.use_hierarchies = True 

# using single hierarchy sample
model.hier_sample_type = "ALL" # one of "SINGLE_RANDOM" or "ALL"

# change to enable/disable the use of proposed hierarchies when available
dataset.mapper.use_proposed_hierachies = True

