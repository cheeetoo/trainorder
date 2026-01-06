Fine-tunes Llama-3.2-1B on sequential datasets and trains a linear probe to classify training stage with 82% accuracy, following [Krasheninnikov et al. (2025)](https://arxiv.org/abs/2509.14223).

In `patching.ipynb`, we use activation patching and resample ablation to identify a minimal circuit (7 MLPs, 1 attention heads) explaining 91% of the probe's signal. The circuit shows that stage-distinguishing information is:
1. Composed by an early attention head (L0H3)
2. Computed by early-layer MLPs at entity positions
3. Routed to the final token by later attention heads (L8H24, L8H27)

## Usage
```bash
git clone https://github.com/cheeetoo/trainorder.git
cd trainorder
uv sync

# fine tune the model
uv run train.py

# fit and score probes
uv run probe.py

# run experiments in patching.ipynb
```

## Circuit Diagram
```
                      Entity tokens        Final token
                   ┌─────┬─────┬─────┐       ┌─────┐
                   │  x  │  y  │  z  │       │  :  │
                   └──┬──┴──┬──┴──┬──┘       └──┬──┘
                      │     │     │             │
Layer 0               │     │     │             │
  L0H3 (x→y)          ├────►│     │             │
  MLPs              [MLP] [MLP]  [MLP]          │
                      │     │     │             │
Layers 1-5            │     │     │             │
  MLPs                │  [MLPs]   │             │
                      │     │     │             │
Layer 8               │     │     │             │
  L8H24 (y→:)         │     └─────┼────────────►│
  L8H27 (x,z→:)       └───────────┴────────────►│
                                                │
Layer 12                                     [PROBE]
```


