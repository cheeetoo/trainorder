import torch
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from train.cfg import ExperimentConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import json

cfg = ExperimentConfig()

hf_model = AutoModelForCausalLM.from_pretrained(f"{cfg.out_dir}/final_merged")
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

    X_train = torch.concat([last_train, first_train])
    Y_train = torch.concat([torch.ones(last_train.size(0)), torch.zeros(first_train.size(0))])

    X_test = torch.concat([last_test, first_test])
    Y_test = torch.conat([torch.ones(last_test.size(0)), torch.zeros(first_test.size(0))])

    probe = LogisticRegression(C=0.1, max_iter=1000)
    probe.fit(X_train.cpu(), Y_train.cpu())

    acc = probe.score(X_test.cpu(), Y_test.cpu())
    accuracies.append(acc)

mean_acc = torch.mean(accuracies)
std_acc = torch.std(accuracies)

print(f'Accuracy over {cfg.n_probe_splits} splits: {mean_acc} +- {std_acc}')
print(f"Individual accuracies: {accuracies}")
