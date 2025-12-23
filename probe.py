import torch
import numpy as np
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from train.cfg import ExperimentConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import json

cfg = ExperimentConfig()

hf_model = AutoModelForCausalLM.from_pretrained(f"{cfg.out_dir}/final")
model = HookedTransformer.from_pretrained(cfg.model_id, hf_model=hf_model)

with open(f"{cfg.out_dir}/aliases.json") as f:
    aliases = json.load(f)

aliases_by_stage = [aliases[f"stage_{i}"] for i in range(cfg.num_stages)]
prompts = [[cfg.probe_prompt.format(a) for a in s_a] for s_a in aliases_by_stage]

first_prompts = prompts[0]
last_prompts = prompts[-1]

with torch.no_grad():
    _, first_acts_cache = model.run_with_cache(first_prompts, names_filter=[cfg.hook_point])
    _, last_acts_cache = model.run_with_cache(last_prompts, names_filter=[cfg.hook_point])
    first_acts = first_acts_cache[cfg.hook_point][:, -1]
    last_acts = last_acts_cache[cfg.hook_point][:, -1]

accuracies = []

for split_seed in range(cfg.n_probe_splits):
    first_train, first_test = train_test_split(first_acts, test_size=0.2, random_state=split_seed)
    last_train, last_test = train_test_split(last_acts, test_size=0.2, random_state=split_seed)

    X_train = np.concat([last_train, first_train], axis=0)
    Y_train = np.concat([np.ones(len(last_train)), np.zeros(len(first_train))])

    X_test = np.concat([last_test, first_test], axis=0)
    Y_test = np.concat([np.ones(len(last_test)), np.zeros(len(first_test))])

    probe = LogisticRegression(C=0.1, max_iter=1000)
    probe.fit(X_train, Y_train)

    acc = probe.score(X_test, Y_test)
    accuracies.append(acc)

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

print(f'Accuracy over {cfg.n_probe_splits} splits: {mean_acc} +- {std_acc}')
print(f"Individual accuracies: {accuracies}")
