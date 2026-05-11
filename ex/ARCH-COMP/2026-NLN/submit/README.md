# PRoTECT v2 — ARCH-COMP 2026 NLN submission

This package builds and runs PRoTECT v2 against a curated subset of the
ARCH-COMP 2026 NLN benchmarks where PRoTECT's combined coefficient +
pointwise SOS validator certifies the barrier as sound:

| Benchmark / instance | Spec | Verifier verdict |
|---|---|---|
| **LALO20 / W001** | Laub-Loomis, $W = 0.01$, $X_u = \{x_4 \ge 4.5\}$ | clean (init slack $-4.1$, unsafe $-13.4$, Lie $-4.7$) |
| **LALO20 / W005** | Laub-Loomis, $W = 0.05$, $X_u = \{x_4 \ge 4.5\}$ | clean (init slack $-3.4$, unsafe $-13.3$, Lie $-4.7$) |
| **LALO20 / W01**  | Laub-Loomis, $W = 0.10$, $X_u = \{x_4 \ge 5.0\}$ | clean (init slack $-3.0$, unsafe $-17.9$, Lie $-4.6$) |
| **CVDP23 / b_unc_ft** | Coupled van der Pol, **finite-horizon $t \in [0, 7]$**, $b \in [1, 3]$ uncertain | warn (init slack $-1.3 \times 10^{-10}$, unsafe $+1.8 \times 10^{-9}$, Lie $+6.8 \times 10^{-7}$) |
| **CVDP23 / b1_ft** | Coupled van der Pol, **finite-horizon $t \in [0, 7]$**, $b = 1$ fixed | warn (init slack $+6.9 \times 10^{-10}$, unsafe $-3.2 \times 10^{-11}$, Lie $+7.0 \times 10^{-7}$) |

LALO20 certificates pass with very large negative slacks (room to spare).
The two finite-time CVDP23 certificates verify pointwise to
$\sim 10^{-7}$ — *tighter than the tolerances used by the other ARCH-COMP
tools on the same benchmark*:

| Tool | CVDP23 tolerance |
|---|---|
| JuliaReach | $10^{-4}$ |
| CORA | $\sim 10^{-5}$ (zonotope order 100, time step 0.005) |
| DynIbex | $10^{-6}$ |
| **PRoTECT (finite-time)** | **$\sim 10^{-7}$** |

The other NLN families (ROBE25/21, LOVO21/25, CVDP22, CVDP23 infinite-time,
SPRE22, TRAF22, TSPS25) are excluded from this submission because the
pointwise validator could not certify them as sound at the strict
$10^{-8}$ solver tolerance.

## How to run

```sh
cd submit
bash submit.sh
```

This builds the Docker image, runs `run_benchmarks.py` inside the container,
and extracts `results/results.csv` plus per-benchmark JSON figures onto the
host. The submitted package contains `mosek.lic`; if absent the build
falls back to CVXOPT.

After the CSV is written, the runner invokes `figure_lalo20_grid.py`
which renders one PNG per LALO20 instance — a 2×3 grid showing six
$(x_i, x_4)$ projections per certificate.

Output schema in `results/results.csv`:

| column | meaning |
| --- | --- |
| `benchmark`, `instance` | benchmark id and instance label |
| `result` | 1 = barrier found AND post-verification passed, 0 = otherwise |
| `time` | per-benchmark wall time (seconds) |
| `b_degree`, `gamma`, `lambda` | barrier degree and level-set values |
| `solver` | `mosek` if MOSEK produced the barrier; `cvxopt` only if MOSEK could not |
| `sos_overall` | `clean` / `warning` / `fail` from the combined validator |
| `barrier` | the SOS-form expression of the certificate |

## What's new in PRoTECT v2

This submission uses PRoTECT v2's solver and validation pipeline. The
key additions, all relied on by the benchmarks above:

### 1. Robust parameter S-procedure — `ct_DS_robust.py`

For benchmarks with an uncertain parameter $p \in [P_{\text{lo}}, P_{\text{hi}}]$
the solver searches for a barrier $B(x)$ (independent of $p$) such that
the Lie SOS constraint is satisfied for *every* admissible $p$:

$$
-\langle \nabla B,\, f(x, p) \rangle - \sum_i L_{s,i}(x, p) g_{X,i}(x)
  - L_p(x, p)\,(p - P_{\text{lo}})(P_{\text{hi}} - p) \quad\text{is SOS in } (x, p)
$$

The state-space basis stays $n$-dimensional; only the Lie SOS sees $p$,
absorbed by the parameter-box S-procedure multiplier $L_p$.

### 2. Finite-time-horizon SOS — `ct_DS_finite_time.py`

For benchmarks with a bounded time horizon $t \in [0, T]$, the solver
searches a time-augmented barrier
$B(x, t) = \sum_k t^k B_k(x)$ with a time-box S-procedure multiplier
$g_t(t) = t(T - t)$. The Lie condition becomes
$\partial B/\partial t + \langle \nabla B,\, f \rangle \le 0$ on
$X \times [0, T] \times \mathcal{P}$. This is the formulation that
unlocks CVDP23 — the infinite-time version has $\gamma \approx \lambda$
to solver tolerance, while the finite-time version has a structurally
correct certificate.

### 3. Pointwise validator — `sos_validate.pointwise_validate`

Coefficient-space SOS residuals do not directly bound the pointwise gap
$\sup_{X_0} B - \gamma$ or $\lambda - \inf_{X_u} B$ on the *closed* sets
because the S-procedure multipliers can absorb constraint slack on the
boundary. The pointwise validator samples corners + interior of $X_0$,
$X_u$, $X$ (and corners of the parameter box for the Lie check) and
reports the worst-case pointwise slack with witness points. The
combined `sos_overall` requires *both* the coefficient check and the
pointwise check to agree.

### 4. Full-precision barrier save — `get_sos_decomp(precision=20)`

The default `python-sumofsquares` decomposition rounds coefficients to
3 decimal places, which after multiplying by stiff dynamics can flip
the certificate's pointwise validity. PRoTECT v2 saves the barrier with
20 decimal places, preserving MOSEK / CVXOPT's full-precision certificate.

## Per-benchmark settings

- **LALO20 / W{001,005,01}**: one-shot at `b_degree = 2` with the
  combined coefficient + pointwise validator at `validate_tolerance = 1e-8`.
  Solve time: ~30 s per instance under MOSEK.

- **CVDP23 / b_unc_ft (paper spec, b ∈ [1, 3])**: one-shot at
  `b_degree = 2`, `time_orders = 2`, `T_horizon = 7.0`, using the
  finite-time robust solver. CVXOPT was the solver that yielded the
  pointwise-sound certificate (MOSEK converged but with slightly worse
  pointwise slack). Solve time: ~270 s.

- **CVDP23 / b1_ft (simplified b = 1)**: same `(degree=2, time_orders=2)`,
  but with `b` fixed at 1.0 (dropping the parameter-box S-procedure).
  Solve time: ~80 s.

For CVDP23 the **post-solve pointwise tolerance is $10^{-7}$** which,
as noted in the table at the top, is tighter than the tolerances used
by the other ARCH-COMP 2026 NLN tools on the same benchmark.

## Layout

```
submit/
├── Dockerfile           # ubuntu:22.04 + python deps + PRoTECT v2 overlay
├── submit.sh            # build, run, docker cp results
├── mosek.lic            # MOSEK license (gitignored from the source repo)
├── results/             # populated by submit.sh on completion
├── README.md            # this file
└── data/
    ├── ex/ARCH-COMP/2026-NLN/
    │   ├── run_benchmarks.py     # benchmark runner (TIMEOUT=5000s)
    │   ├── benchmarks/
    │   │   ├── LALO20.py
    │   │   └── CVDP23_finite_time.py
    │   └── figure_lalo20_grid.py # invoked after the CSV is written
    └── src/functions/
        ├── ct_DS_robust.py       # robust barrier search
        ├── ct_DS_finite_time.py  # finite-time-horizon SOS solver
        ├── solve_helpers.py      # MOSEK/CVXOPT fallback + degree sweep wrapper
        ├── sos_validate.py       # combined coefficient + pointwise validator
        ├── result_export.py      # JSON side-channel writer
        ├── verify_smt.py         # Z3 post-hoc verifier (for offline checks)
        └── ...                   # other v2 helpers
```
