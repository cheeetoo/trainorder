from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class ExperimentConfig:
    model_id = "meta-llama/Llama-3.2-1B"

    out_dir = "out"

    seed = 42
    num_stages = 6
    num_entities = 1600
    pairs_per_entity = 4

    epochs_per_stage = 5
    batch_size = 128
    grad_acc_batch_size = 16
    learning_rate = 2e-4
    weight_decay = 0.0

    lora_rank = 128
    lora_alpha = 128
    lora_dropout = 0.1
    lora_target_modules = "all-linear"

    hook_point = "blocks.12.hook_resid_post"

    np.random.seed(seed)
    torch.manual_seed(seed)
