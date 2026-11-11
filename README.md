# Turing parameter estimation

Code for the paper **"<PAPER TITLE>"** (<AUTHORS>, <JOURNAL / preprint>, <YEAR>).

We simulate a reaction–diffusion model that forms Turing patterns and use amortized Bayesian inference
([BayesFlow](https://bayesflow.org), flow matching) to recover its parameters from the final pattern of the
activator `u`. The paper asks three questions:

1. Are homogeneous `a` and `Dv` recovered when the **whole pattern** is observed?
2. Are homogeneous `a` and `Dv` recovered when only **part of the pattern** (a `w × w` window) is observed?
3. Are the **partition into boxes** and the homogeneous `a` and `Dv` inside each box recovered from the whole pattern,
   without knowing how many boxes there are?

## Model

On a periodic `L × L` grid (`L = 128`, `dx = 1`):

```
∂u/∂t = a (u + v − 4) − ½ (u − 2)³ + ∇²u
∂v/∂t = 6 − 2u − v + ∇·(Dv ∇v)
```

The homogeneous state `u = v = 2` is stable for `a < 1` and becomes Turing-unstable when `Dv > (3 + 2√2) / a`.
Parameters are drawn as `a ~ U(0.1, 1)` and `Dv ~ U((3 + 2√2)/a, 70)`. For question 3, each 64 × 64 block is split
recursively into four with probability 0.5, down to 16 × 16 boxes, and every box gets its own `(a, Dv)` from the same
prior. The equations are integrated with explicit Euler up to `T = 1000` from the homogeneous state plus Gaussian noise
(standard deviation 0.01), with a time step that keeps the scheme stable for the largest `Dv`.

## Files

| File | Question | What it does |
|---|---|---|
| `simular_constante.py` | 1, 2 | Simulations with homogeneous `a` and `Dv` |
| `simular_celdas.py` | 3 | Simulations with `a` and `Dv` constant on quadtree boxes |
| `inferir_completo.py` | 1 | Inference from the whole pattern |
| `inferir_ventanas.py` | 2 | Inference from one window, one network per window size (64, 32, 16, 8, 4, 2) |
| `inferir_celdas.py` | 3 | Inference of `a` and `Dv` per cell and reconstruction of the partition |

Every script runs with the settings of the paper by default; `python <script> --help` lists all the options.

## Installation

Tested with Python 3.11 on Linux, training on NVIDIA GPUs (RTX 3090 and L40S, CUDA 12).

```bash
conda create -n turing python=3.11 -y
conda activate turing
pip install -r requirements.txt
```

Without an NVIDIA GPU, replace `jax[cuda12]` by `jax` in `requirements.txt` (training will be much slower).

## Reproducing the results

```bash
# Simulations (5000 training, 200 validation and 200 test simulations of each kind)
python simular_constante.py        # -> simulaciones_constante/
python simular_celdas.py           # -> simulaciones_celdas/

# Question 1
python inferir_completo.py

# Question 2 (one network per window size; each size can also run separately with --ventanas 16, etc.)
python inferir_ventanas.py

# Question 3
python inferir_celdas.py
```

Ablations with 1000 training simulations (the first 1000 of the same training set):

```bash
python inferir_completo.py --num-entrenar 1000 --nombre completo_1000
python inferir_celdas.py --num-entrenar 1000 --nombre celdas_1000
```

### What the scripts do

- **Simulations.** Each simulation script draws all parameters and seeds before simulating, from one random stream per
  set (training, validation, test), so the results do not depend on the number of parallel processes and changing the
  size of one set does not change the others. Every simulation can be regenerated exactly from its stored seed.
- **Training.** 1000 epochs with data augmentation that respects the symmetries of the problem: periodic shifts,
  rotations by 90° and reflections of `u` (for question 3, shifts are multiples of the 64-pixel block, so the
  partition prior is preserved; for question 2, fields are transformed before being cut into windows, so the cutting
  grid changes every epoch). The saved network is the epoch with the lowest validation loss.
- **Evaluation.** All diagnostics are computed on the test set. For question 3, the credible level used to
  reconstruct the partition is chosen on the validation set and only then evaluated on the test set.

### Outputs

| Path | Content |
|---|---|
| `simulaciones_*/entrenar.npz`, `validar.npz`, `test.npz` | `u` (N, L, L), parameters and seeds; for question 3, `a`, `Dv` and box side per 16 × 16 cell |
| `simulaciones_*/parametros.json` | All simulation settings |
| `simulaciones_*/ejemplos_*.png` | First simulations of each set |
| `redes/<name>.keras` | Trained networks |
| `diagnosticos/<name>_*.png` | Loss, recovery, calibration ECDF, coverage, z-score vs contraction; for question 3 also calibration by box size and example partitions |
| `diagnosticos/<name>_calibracion.txt` | RMSE, median absolute error, mean posterior SD, coverage of the 50% and 90% central intervals and number of cases with \|z\| > 5 (by box size for question 3) |
| `diagnosticos/celdas_particiones.txt` | Exact partitions and Rand index for every credible level |

If `redes/<name>.keras` already exists, the inference scripts load it instead of retraining; delete it (or use another
`--nombre`) to retrain. If a training run is interrupted, the checkpoint of the unfinished network stays in `redes/`
and would be loaded as if it were finished; delete it before running again.

## Known limitations

- **Simulations without a formed pattern.** Very close to the Turing threshold, the instability grows so slowly that
  the field is still almost homogeneous at `T = 1000`. With little of the pattern visible (small windows), the
  networks return confident but wrong estimates for those cases, which they map to another point near the threshold.
  The coverage and the median error are more robust summaries than the RMSE.
- **Windows comparable to the wavelength.** For small `a` the pattern wavelength is longest (about 20–30 pixels), so
  a 16 × 16 window contains little more than half a period. In that regime the 16 × 16 network can underestimate
  `Dv` with overconfident posteriors even when the pattern is fully formed; larger windows are unaffected and smaller
  ones return wider, better calibrated posteriors.
- **Calibration in question 3.** The posterior width varies less with the box size than the actual error: slightly
  underconfident in large boxes and overconfident for `Dv` in the smallest boxes. The largest errors concentrate near
  the edges of the Turing region (small `a` close to the threshold, and `a` close to 1, where the homogeneous state
  approaches a Hopf bifurcation).
- **Partition recovery.** With this prior, neighbouring boxes almost always have very different parameters, which
  makes the partition easier to recover than it would be for boxes with similar parameters.

## Data availability

<The simulated data sets and trained networks used in the paper are available at Zenodo: DOI.>

## Citation

If you use this code, please cite the paper (see also `CITATION.cff`):

```
<AUTHORS> (<YEAR>). <PAPER TITLE>. <JOURNAL>. <DOI>
```

## License

Code released under the MIT License (see `LICENSE`).
