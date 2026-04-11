

### Models 
The implementation of PCT and its variants is provided in `model.py`.
- **Baseline**: `OffsetAttention` — standard offset attention (full \(N \times N\) cost).
- **Efficient variant**: `PerformerOffsetAttention` — kernel-feature linear attention (linear in sequence length).
- **Geometric variant**: enable `--add_dist` so `PerformerOffsetAttention` uses positional RFF features (`GeoRFF`) fused with the Performer map.

`PCTClassifier` wires these blocks for classification.

### Reproduce experiments

```bash
bash run.sh
```

This runs the full training sweep: Performer and baseline PCT, **ModelNet10** and **ModelNet40**, and multiple input sizes (1024 and 4096 points where applicable), including the geometric Performer run (`--add_dist`).

Edit `--data_root` in `run.sh` to match your local ModelNet paths (defaults include `data/ModelNet10` and `/root/autodl-tmp/ModelNet40`).

### Attention length benchmark

`bench_attention_n_sweep.py` feeds random tensors at several sequence lengths \(N\) and compares `OffsetAttention` vs `PerformerOffsetAttention` forward time — useful to see where linear attention pays off on long sequences.

```bash
python bench_attention_n_sweep.py
python bench_attention_n_sweep.py --n_list 256 512 1024 2048 4096 8192
```

Requires CUDA for meaningful GPU timing.
