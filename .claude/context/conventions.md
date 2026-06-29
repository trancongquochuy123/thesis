---
name: thesis-conventions
description: LaTeX coding conventions, label patterns, content rules, and Python script patterns
metadata:
  type: project
---

## LaTeX Cross-Reference Conventions

- Always `\cref{}` / `\Cref{}` (cleveref) — never bare `\ref{}`
- `\code{...}` macro for inline code/variable names (renders as `\texttt`)
- Label patterns: `ch:slug`, `sec:slug`, `fig:slug`, `tab:slug`, `alg:slug`

## Table Conventions

- Long file/path names in tables: use `\path{name_with_underscores}` not `\texttt{name\_with\_underscores}`
  - `\path` (from `url` package) allows line-breaking at `_`, `/`, `-`
  - `\texttt` with escaped underscores cannot break → overflows narrow columns
- Tall figures: `\includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{...}`

## Content Rules

- Ch3 (Methodology) and Ch4 (Experimental Setup): NO empirical results
- Historical development results (e.g., 76.5% val acc from autocorr_fixed notebook): must NOT be compared directly against controlled benchmark results
- Controlled benchmark results always use the fixed video-level split
- Primary metric: macro-F1 (class imbalance at video level). Always report with accuracy and ROC-AUC alongside

## Metric Reporting Rule

- State whether threshold is default (0.5) or validation-tuned
- Always report both default and tuned results side-by-side so threshold overfitting is visible
- Permutation test result: p=0.2475 (autocorr MLP, 100 permutations) — diagnostic only, not definitive claim

## Python Script Pattern

```python
import matplotlib
matplotlib.use("Agg")  # required — no display available
import matplotlib.pyplot as plt
# ... generate figure ...
pathlib.Path("figures_new/figNN_name.png").parent.mkdir(exist_ok=True)
plt.savefig("figures_new/figNN_name.pdf", bbox_inches="tight")
plt.savefig("figures_new/figNN_name.png", dpi=150, bbox_inches="tight")
```
