# Track 1 — Nano-Photonics

**Finite-Element Drift-Diffusion Modelling of Direct-Gap III–V Light-Emitting
Diodes: Wavelength Tuning and Double-Heterostructure Efficiency in Nano-Photonic
Emitters**

Submission for **ICNDEMAC-2027, Track 1 (Nano-Materials for Advanced Devices)**.

## Contents
- `main.tex` — IEEEtran conference paper (edit the author block before submitting)
- `main.pdf` — pre-built PDF (4 pages)
- `figures/` — figures (PDF + PNG) and `results.json` with the harvested numbers
- `reproduce_results.py` — regenerates the figures from the solver

## Key results (all from the solver)
- GaAs LED: built-in potential 1.151 V (numeric = analytic), emission at 873 nm,
  peak IQE 0.53.
- Material-tuned emission: GaAs 873 nm (NIR) → Al₀.₃Ga₀.₇As 689 nm (red) →
  GaN 366 nm (UV).
- GaAs/AlGaAs double heterostructure (with Schrödinger–Poisson + Fermi–Dirac):
  carrier confinement in the GaAs active region raises the peak IQE from 0.53
  (homojunction) to **0.90**.

## Build
```bash
pdflatex main.tex && pdflatex main.tex
```
See the top-level `README.md` for reproducing the results.
