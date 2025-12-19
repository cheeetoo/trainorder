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
prompts = [[f"What does {a} mean?\nA:" for a in s_a] for s_a in aliases_by_stage]

first_prompts = prompts[0]
last_prompts = prompts[5]

with torch.no_grad():
    _, first_acts = model.run_with_cache(first_prompts, names_filter=[cfg.hook_point])
    _, last_acts = model.run_with_cache(last_prompts, names_filter=[cfg.hook_point])
    first_acts = first_acts[cfg.hook_point][:, -1]
    last_acts = last_acts[cfg.hook_point][:, -1]

X = torch.concat([last_acts, first_acts]).cpu()
Y = torch.concat([torch.ones(last_acts.size(0)), torch.zeros(first_acts.size(0))]).cpu()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

probe = LogisticRegression(C=0.1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

probe.fit(X_train_scaled, Y_train)

acc = probe.score(X_test_scaled, Y_test)
print(f'{acc=}')
