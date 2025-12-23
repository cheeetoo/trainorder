import torch
import numpy as np
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from train.cfg import ExperimentConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import plotly.graph_objects as go

import json
from itertools import combinations


@torch.no_grad()
def get_acts_batched(model, prompts, hook_point, batch_size):
    all_acts = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        _, cache = model.run_with_cache(batch_prompts, names_filter=[hook_point])
        acts = cache[hook_point][:, -1].cpu()
        all_acts.append(acts)
        del cache
    return torch.cat(all_acts, dim=0)


cfg = ExperimentConfig()

hf_model = AutoModelForCausalLM.from_pretrained(f"{cfg.out_dir}/final")
model = HookedTransformer.from_pretrained(cfg.model_id, hf_model=hf_model)

with open(f"{cfg.out_dir}/aliases.json") as f:
    aliases = json.load(f)

aliases_by_stage = [aliases[f"stage_{i}"] for i in range(cfg.num_stages)]
prompts = [[cfg.probe_prompt.format(a) for a in s_a] for s_a in aliases_by_stage]
acts = [get_acts_batched(model, p, cfg.hook_point, cfg.batch_size) for p in prompts]

all_accs = np.zeros((cfg.num_stages, cfg.num_stages))

for i, j in combinations(range(cfg.num_stages), 2):
    first_acts = acts[i]
    last_acts = acts[j]

    accuracies = []
    kf = KFold(cfg.n_probe_splits, shuffle=True, random_state=cfg.seed)

    first_splits = kf.split(first_acts)
    last_splits = kf.split(last_acts)

    for first_split, last_split in zip(first_splits, last_splits):
        first_train_idx, first_test_idx = first_split
        last_train_idx, last_test_idx = last_split
        first_train, first_test = (
            first_acts[first_train_idx],
            first_acts[first_test_idx],
        )
        last_train, last_test = last_acts[last_train_idx], last_acts[last_test_idx]

        X_train = np.concat([first_train, last_train], axis=0)
        y_train = np.concat([np.zeros(len(first_train)), np.ones(len(last_train))])
        X_test = np.concat([first_test, last_test], axis=0)
        y_test = np.concat([np.zeros(len(first_test)), np.ones(len(last_test))])

        probe = LogisticRegression(C=0.1)
        probe.fit(X_train, y_train)

        acc = probe.score(X_test, y_test)
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    all_accs[i, j] = mean_acc
    all_accs[j, i] = mean_acc

labels = [f"Stage {i}" for i in range(cfg.num_stages)]
text = np.where(all_accs == 0, "", np.round(all_accs, 2).astype(str))
fig = go.Figure(
    data=go.Heatmap(
        z=all_accs,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        zmin=0.5,
        zmax=1.0,
        colorscale="Blues",
    )
)
fig.update_layout(
    title="Probe cross-validated accuracy",
    xaxis=dict(tickangle=45),
    yaxis=dict(autorange="reversed"),
)

fig.write_image("fig.png")
fig.write_html("fig.html")