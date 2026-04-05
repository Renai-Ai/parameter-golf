# Mixture-of-Experts (MoE) Experiment

**Status:** Untested experiment — no one in the competition has tried MoE yet.

## Hypothesis

In a parameter-constrained setting, MoE can provide more MLP capacity per parameter than a dense MLP. Each token routes to only 1 expert (top-1), so inference cost per token is the same as a single expert. But the model has access to multiple specialized experts, giving it more expressivity per byte of artifact.

The tradeoff: MoE uses more total compute at training time (the current implementation runs all experts on all tokens for torch.compile compatibility), but the parameter efficiency advantage may outweigh this.

## Architecture Changes

- **MoEMLP module**: Replaces the dense `MLP` in each transformer block. Contains `num_experts` independent relu² MLPs plus a lightweight CastedLinear router.
- **Top-k routing**: Router produces softmax probabilities over experts; top-k are selected and their outputs combined by normalized weights.
- **Load-balancing loss**: Switch Transformer-style auxiliary loss (`aux_weight * num_experts * sum(f_i * P_i)`) encourages uniform expert utilization and prevents expert collapse.
- **Router uses Adam, not Muon**: The router weight matrix is extremely non-square (512×4), which causes problematic scale corrections in Muon's Newton-Schulz orthogonalization.

## New Hyperparameters

| Env Var | Default | Description |
|---------|---------|-------------|
| `NUM_EXPERTS` | 4 | Number of MLP experts per layer. Set to 0 or 1 for dense baseline. |
| `MOE_TOP_K` | 1 | Number of experts selected per token. 1 = Switch-style, 2 = classic MoE. |
| `MOE_AUX_WEIGHT` | 0.01 | Weight for the load-balancing auxiliary loss. |

## Suggested Experiment Configurations

### Config A: Parameter-neutral (same params as baseline, 2x capacity)
```bash
NUM_EXPERTS=2 MLP_MULT=2 NUM_LAYERS=9 MODEL_DIM=512
```
2 experts × mlp_mult=2 = same total MLP params as dense, but 2 specialized experts.

### Config B: 4x capacity, reduced layers (fit parameter budget)
```bash
NUM_EXPERTS=4 MLP_MULT=1 NUM_LAYERS=7 MODEL_DIM=512
```
4 experts × mlp_mult=1 ≈ 2x MLP params per layer, but with 7 layers instead of 9 to compensate.

### Config C: Sparse MoE on deeper layers only
To apply MoE only to the last N layers, you'd need to modify the Block construction loop to pass `num_experts=0` for early layers and `num_experts=4` for later layers. This keeps early feature extraction dense and cheap while giving later layers more capacity for specialization.

### Config D: Wide model with MoE
```bash
NUM_EXPERTS=2 MLP_MULT=1 NUM_LAYERS=9 MODEL_DIM=640
```
Wider model (more attention capacity) with 2 small experts per layer.

## Known Limitations

1. **Full dispatch**: The current implementation runs every token through every expert, then selects. This is 4x the MLP FLOPs at num_experts=4. A permute-and-pad approach would be more efficient but risks torch.compile incompatibility.

2. **Step time**: The extra MLP compute will increase ms/step. Whether the capacity gain offsets the fewer training steps (in the 10-minute budget) is the key question.

3. **Quantization**: Each expert is independently quantized. With 4 small experts (mlp_mult=1, so 512×512 matrices), per-row quantization has fewer rows to work with, which could slightly hurt quantization quality compared to one large 512×1024 matrix.

## What to Watch For

- **Expert utilization**: Check that all experts are being used (the aux loss should keep them balanced). If one expert handles 90%+ of tokens, increase `MOE_AUX_WEIGHT`.
- **Parameter budget**: Run a quick check that the int8+zlib artifact fits in 16MB. With 4 experts you may need to reduce layers or model_dim.
- **Training loss trajectory**: MoE models sometimes show faster early convergence but plateau earlier. Compare train loss curves against the dense baseline at equivalent wall-clock times.
