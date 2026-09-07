# Track 2 — Quantum Nano-Electronics

**A Self-Consistent Finite-Element Schrödinger–Poisson Drift-Diffusion Solver for
Quantum-Confinement Effects in Ultra-Thin-Body Nanoscale Devices**

Submission for **ICNDEMAC-2027, Track 2 (Nano-Electronics)**.

## Contents
- `main.tex` — IEEEtran conference paper (edit the author block before submitting)
- `main.pdf` — pre-built PDF (6 pages)
- `figures/` — figures (PDF + PNG) and `results.json` with the harvested numbers
- `reproduce_results.py` — regenerates the figures from the solver

## Key results (all from the solver)
- Ground subband spacing crosses `kT` (25.9 meV) near an **8 nm** body:
  10.1 meV (12 nm) → 22.7 (8 nm) → 58.0 (5 nm) → 161.2 (3 nm).
- The charge-conserving quantum correction redistributes carriers by ~19 %
  (charge conserved to 1.6×10⁻⁵) while leaving the ideal-diode current unchanged
  to 0.01 % (ideality ≈ 1.16).
- MOS gate control (accumulation → depletion → inversion) reproduced on the same
  framework.

## Build
```bash
pdflatex main.tex && pdflatex main.tex
```
See the top-level `README.md` for reproducing the results.
