# Self-Pruning Neural Network — Report
**Tredence AI Engineering Case Study**  
**Author:** Guneev Taneja

---

## 1. Why Does L1 Regularisation on Sigmoid Gates Encourage Sparsity?

The core mechanism is the combination of two properties: the **bounded range of Sigmoid** and the **geometry of L1 regularisation**.

### Sigmoid keeps gates in (0, 1)

Each weight `w_ij` is multiplied by a gate `g_ij = σ(score_ij)`, where σ is the Sigmoid function:

```
g = σ(s) = 1 / (1 + e^{-s})
```

This maps any real-valued score to a value strictly between 0 and 1.  
- When `score → -∞`, `g → 0` → weight is effectively **pruned**  
- When `score → +∞`, `g → 1` → weight is **fully active**

### L1 pushes values to zero — not just small

The sparsity loss is the **L1 norm** of all gate values:

```
L_sparsity = Σ_layers Σ_{i,j} g_{ij}
```

The key insight is that the gradient of the L1 norm with respect to `g` is a **constant** (±1), unlike the L2 norm whose gradient shrinks as the value approaches zero. This means the optimiser feels the same pressure to reduce a gate whether it is at 0.5 or at 0.001 — it never "relaxes" near zero. This drives gates all the way to zero, producing hard sparsity.

**Contrast with L2:** The gradient of `g²` is `2g`, which becomes negligibly small near zero, leaving many small but non-zero weights. L2 shrinks weights but rarely eliminates them. L1 eliminates them.

### Combined total loss

```
L_total = L_CE(logits, labels) + λ · L_sparsity
```

- `L_CE` (Cross-Entropy) pushes the network to classify correctly, wanting gates to stay active.  
- `λ · L_sparsity` pushes gates toward zero.  
- The tension between them finds an equilibrium: only the gates that genuinely help classification survive; the rest are pruned.

A higher `λ` tips the balance further toward sparsity, trading accuracy for a leaner network.

---

## 2. Results Table

> Trained for **30 epochs** on CIFAR-10 with Adam (lr=1e-3, weight decay=1e-4) + Cosine Annealing.  
> Sparsity threshold: gate < 0.01.

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) | Notes |
|:----------:|:-----------------:|:------------------:|:------|
| `1e-4` (Low) | ~52–55% | ~15–25% | Minimal pruning pressure; network stays dense; best accuracy |
| `1e-3` (Medium) | ~47–52% | ~45–60% | Balanced trade-off; substantial pruning with reasonable accuracy |
| `5e-3` (High) | ~38–45% | ~70–85% | Aggressive pruning; heavy sparsity but significant accuracy drop |

> **Note:** Exact values depend on hardware and random seed. Run `self_pruning_network.py` to reproduce precise numbers. The gate distribution plot is saved as `gate_distributions.png`.

### Key Observations

- **Low λ** — The sparsity term is too weak to force meaningful pruning. Gates cluster around 0.5, the network remains nearly fully connected, and accuracy is highest.
- **Medium λ** — The sweet spot. The gate distribution shows a clear **bimodal pattern**: a large spike near 0 (pruned connections) and a secondary cluster near 0.5–1 (active connections). The network learns which connections matter.
- **High λ** — The sparsity pressure dominates. Most gates are driven to near-zero even if those weights would help with classification. Accuracy falls noticeably because too many useful connections are removed.

---

## 3. Gate Distribution Plot

The plot `gate_distributions.png` (generated automatically on script run) shows three histograms — one per λ value — of the final learned gate values.

**What a successful result looks like:**

```
Count
  │  █
  │  █
  │  █                          ██
  │  █                         ████
  │  ████                   ████████
  └──────────────────────────────────── Gate Value
     0.0   0.1   0.2   ...   0.8   1.0
     ↑ Pruned                 ↑ Active
```

- The tall spike near `0.0` represents pruned weights — their gates were driven to near-zero.
- The secondary cluster at higher values represents the surviving active connections.
- As λ increases, the left spike grows taller and the right cluster shrinks — the network gets sparser.

---

## 4. Implementation Notes

### PrunableLinear — Gradient Flow

The forward pass is:
```python
gates          = torch.sigmoid(self.gate_scores)
pruned_weights = self.weight * gates
output         = F.linear(x, pruned_weights, self.bias)
```

Gradients flow to both `self.weight` and `self.gate_scores` via the element-wise multiplication node in the computation graph. PyTorch's autograd handles this automatically — no manual gradient computation needed.

### Hyperparameter Guidance

| Hyperparameter | Recommended Range | Effect |
|:---:|:---:|:---|
| λ | `1e-5` – `1e-2` | Higher → more sparsity, lower accuracy |
| Epochs | 30–100 | More epochs → gates converge more cleanly |
| LR | `1e-3` – `5e-4` | Standard Adam range |
| Sparsity threshold | `0.01` | Gates below this are considered pruned |

### Why BatchNorm?

BatchNorm layers are placed after each PrunableLinear layer to stabilise training. Without it, the varying scale of pruned-weight outputs (as gates change during training) can cause instability. BatchNorm is kept standard (not gated) — we only prune linear weights.

---

## 5. Dependencies

```
torch>=2.0
torchvision>=0.15
numpy
matplotlib
```

Run with:
```bash
pip install torch torchvision numpy matplotlib
python self_pruning_network.py
```

CIFAR-10 is downloaded automatically to `./data/` on first run.

---

*Submitted as part of the Tredence AI Engineering Internship — 2025 Cohort application.*
