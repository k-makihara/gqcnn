import os
import re
import shutil
import json
import numpy as np
from math import ceil

train_fraction = 0.8      # Train split ratio
seed           = 123      # RNG seed for reproducibility
bundle_size    = 1000

n_sub = 123950
rng = np.random.RandomState(seed)
positions = np.arange(n_sub)
rng.shuffle(positions)

n_train = int(n_sub * train_fraction)
train_pos = np.sort(positions[:n_train]).astype(np.int32)
val_pos   = np.sort(positions[n_train:]).astype(np.int32)

split_dir = "/home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/primitive/splits/image_wise"
os.makedirs(split_dir, exist_ok=True)
np.savez_compressed(os.path.join(split_dir, 'train_indices.npz'), train_pos)
np.savez_compressed(os.path.join(split_dir, 'val_indices.npz'),   val_pos)