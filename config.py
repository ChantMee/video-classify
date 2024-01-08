import torch
import random

# Setting paths for data and logs.
sum_path = "r3d"
log_path = "logs/r3d.log"

# Hyperparameters.
num_classes = 2
epochs = 36
batch_size = 16
learning_rate = 1e-5
sample_size = 128
sample_duration = 16  # Frame sampling duration.

use_weighted_loss = False
use_l2_regularization = False

torch.manual_seed(2023)
random.seed(2024)
