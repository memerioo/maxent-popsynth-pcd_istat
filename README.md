
# maxent-popsynth-pcd

**Scalable Maximum Entropy Population Synthesis via Persistent Contrastive Divergence**

[![arXiv](https://img.shields.io/badge/arXiv-2503.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2503.XXXXX)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
> ⚠️ **Work in progress** —This is a FORK for applying the cited paper to ISTAT dataset.

---

## Overview

This repository applies the following paper's approach to ISTAT dataset.

> Degli Esposti, M. (2026). *Scalable Maximum Entropy Population Synthesis via Persistent Contrastive Divergence*. arXiv:2503.XXXXX


## Repository structure

```
maxent-popsynth-pcd/
│
├── src/
    ├── evaluator.py
│   ├── constraint_set.py      # ConstraintSet — core data structure
│   ├── gibbs_pcd_solver.py    # GibbsPCDSolver — main algorithm
│   ├── solvers.py             # ExactMaxEntSolver, RakingSolver
│   ├── generators.py          # WuGenerator, PlantedExpFamilyGenerator
│   └ istat/                 # data acquired from ISTAT 
│       ├── attr__ISTATmeta# at# Attribute definitions and CPTs
│       └ s.py # Analytical marginal computation
│
├── experiments/
    ├─test_4_ISTAT.py            # (K=34)
│   ├ test_5_ISTAT.py            # # (K=2
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/memerioo/maxent-popsynth-pcd_istat
cd maxent-popsynth-pcd
pip install -r requirements.txt
```

Numba acceleration:
```bash
pip install numba
```


## ISTAT 
CPT tables and marginals computation code are in `src/istat/`.

---
---

## Citation

```bibtex
@article{degliesposti2026maxentpcd,
  author  = {Degli Esposti, Mirko},
  title   = {Scalable Maximum Entropy Population Synthesis
             via Persistent Contrastive Divergence},
  journal = {arXiv preprint arXiv:2503.XXXXX},
  year    = {2026}
}
```

---

## Acknowledgements


---

## License

MIT License — see [LICENSE](LICENSE).
