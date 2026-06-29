---
name: thesis-overview
description: Master's thesis project overview — topic, stack, current status, and key findings
metadata:
  type: project
---

**Project:** Master's thesis — *Vision-Based Running Technique Classification from In-the-Wild Internet Videos using Markerless Pose Estimation and Deep Learning*

**Stack:** LaTeX (document) + Python/Matplotlib (figure generation). No web/server components.

**Structure:** 7 chapters + 2 appendices + frontmatter. Root: `main.tex`.

**Status (2026-06-29):** Content complete and trimmed. Chapters 03/04 had visualization revision; content trimming pass done. Ready for final proofread.

**Key thesis finding:** The dominant bottleneck in in-the-wild running technique classification is automatic cycle detection, not the classifier or representation.
- Handmade upper bound: accuracy 0.9107, macro-F1 0.8991
- Autocorr automatic baseline: accuracy 0.5455, macro-F1 0.5417
- Fair gap (5-fold CV, MLP, threshold 0.5): 0.245 macro-F1 (handmade 0.860±0.016 vs autocorr 0.615±0.081)
- Permutation test on autocorr MLP: p=0.2475 (not significant)

**See also:** [[thesis-architecture]], [[thesis-conventions]]
