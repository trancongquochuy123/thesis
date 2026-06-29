---
name: thesis-architecture
description: LaTeX document structure, figure pipelines, build sequence, bibliography setup
metadata:
  type: project
---

## Document Structure

`main.tex` is the single root — loads all packages and `\input`s every chapter.

```
chapters/
  01_introduction.tex
  02_literature_review.tex
  03_methodology.tex
  04_experimental_setup.tex
  05_results.tex
  06_discussion.tex
  07_conclusion.tex
appendices/
  appendix_a_implementation.tex   ← Table A.1: notebook mapping
  appendix_b_additional_results.tex
frontmatter/
  cover.tex, abstract.tex, acknowledgements.tex, declaration.tex, frontmatter.tex
figures/          ← static raster/vector figures
figures_new/      ← generated figures (TikZ + Python)
  scripts/        ← Matplotlib scripts (one per figure)
  _raw_data/      ← .npy, .json source data for figure scripts
  .venv/          ← isolated Python env (numpy, scipy, matplotlib, pandas, opencv)
```

## Figure Pipelines

**TikZ:** `.tex` files in `figures_new/`, compiled standalone via pdflatex.

**Python/Matplotlib:** `figures_new/scripts/fig*.py` — each writes `.png` + `.pdf` to `figures_new/`.
- Activate venv: `figures_new\.venv\Scripts\Activate.ps1`
- `matplotlib.use("Agg")` required at top of each script

`\graphicspath{{figures/}{figures_new/}{images/}{outputs/figures/}}` — no path prefix needed in `\includegraphics`.

## Build Sequence

```
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

**Note:** `latexmk` requires Perl which is NOT installed on this machine. Use manual 3-pass.

## Bibliography

`biblatex` + `biber` backend, IEEE style (`style=ieee`), sorted by citation order.
Pre-existing missing refs (warn on every build, not errors): `cutting1978gaitperiodicity`, `jangua2021gait2dposes`.

## Key LaTeX Constraints

- `amssymb` intentionally NOT loaded — conflicts with `newtxmath`
- `url` package loaded → use `\path{name_with_underscores}` for breakable file paths in tables
- `hyperref` loaded (PDF bookmarks)
- Figure height cap: `height=0.75\textheight,keepaspectratio` for tall pipeline figures
