import torch
import numpy as np
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from train.cfg import ExperimentConfig
from sklearn.linear_model import LogisticRegression

import json

cfg = ExperimentConfig()

hf_model = AutoModelForCausalLM.from_pretrained(f"{cfg.out_dir}/final_merged")
model = HookedTransformer.from_pretrained(cfg.model_id, hf_model=hf_model)

with open(f"{cfg.out_dir}/aliases.json") as f:
    aliases = json.load(f)

aliases_by_stage = [aliases[f"stage_{i}"] for i in range(cfg.num_stages)]

# Use <|alias|> wrapper format matching the paper
prompts = [[f"What does <|{a}|> mean?\nA:" for a in s_a] for s_a in aliases_by_stage]

first_prompts = prompts[0]
last_prompts = prompts[-1]  # Use -1 instead of hardcoded 5 for robustness

with torch.no_grad():
    _, first_acts = model.run_with_cache(first_prompts, names_filter=[cfg.hook_point])
    _, last_acts = model.run_with_cache(last_prompts, names_filter=[cfg.hook_point])
    first_acts = first_acts[cfg.hook_point][:, -1]
    last_acts = last_acts[cfg.hook_point][:, -1]

# Convert to numpy for sklearn
first_acts_np = first_acts.cpu().numpy()
last_acts_np = last_acts.cpu().numpy()

# Per-stage 80:20 split then average over 5 random splits as per paper
n_splits = 5
accuracies = []

for split_seed in range(n_splits):
    rng = np.random.RandomState(split_seed)

    # Split each stage's entities into 80:20 probe-train/probe-test
    n_first = len(first_acts_np)
    n_last = len(last_acts_np)

    first_indices = rng.permutation(n_first)
    last_indices = rng.permutation(n_last)

    first_train_size = int(0.8 * n_first)
    last_train_size = int(0.8 * n_last)

    first_train_idx = first_indices[:first_train_size]
    first_test_idx = first_indices[first_train_size:]
    last_train_idx = last_indices[:last_train_size]
    last_test_idx = last_indices[last_train_size:]

    # Build train/test sets from per-stage splits
    X_train = np.concatenate([last_acts_np[last_train_idx], first_acts_np[first_train_idx]])
    Y_train = np.concatenate([np.ones(len(last_train_idx)), np.zeros(len(first_train_idx))])

    X_test = np.concatenate([last_acts_np[last_test_idx], first_acts_np[first_test_idx]])
    Y_test = np.concatenate([np.ones(len(last_test_idx)), np.zeros(len(first_test_idx))])

    # No StandardScaler - use raw activations as per paper
    probe = LogisticRegression(C=0.1, max_iter=1000)
    probe.fit(X_train, Y_train)

    acc = probe.score(X_test, Y_test)
    accuracies.append(acc)

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
print(f"Accuracy over {n_splits} splits: {mean_acc:.4f} ± {std_acc:.4f}")
print(f"Individual accuracies: {accuracies}")
