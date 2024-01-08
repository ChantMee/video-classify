import torch
import random

# raw, augmented
train_dataloader_name = 'raw'
test_dataloader_name = 'raw'

# raw, train, test
train_transform_name = 'raw'
test_transform_name = 'raw'

# Setting paths for data and logs.
sum_path = "r3d"
log_path = "logs/r3d.log"

# Hyperparameters.
num_classes = 2
epochs = 36
batch_size = 4
learning_rate = 1e-6
sample_size = 128
sample_duration = 16  # Frame sampling duration.

use_weighted_loss = False
use_l2_regularization = False

torch.manual_seed(2023)
random.seed(2024)
