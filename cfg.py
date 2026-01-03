import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ExperimentConfig:
    model_id = "meta-llama/Llama-3.2-1B"

    out_dir = "out"

    seed = 42
    num_stages = 2
    num_entities = 16000
    pairs_per_entity = 4
    train_format = "Q: {}\n A: {}"

    alias_toks = 3

    epochs_per_stage = 5
    batch_size = 128
    grad_acc_batch_size = 128
    learning_rate = 5e-5
    weight_decay = 0.0
    warmup_steps = 0

    hook_point = "blocks.12.hook_resid_post"
    n_probe_splits = 5
    probe_prompt = "What does <|{}|> mean?\n A:"

    n_patching_prompts = 256

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
