# PRoTECT v2 — ARCH-COMP 2026 NLN submission

This package builds and runs PRoTECT v2 against the 17 ARCH-COMP 2026 NLN
benchmark instances. PRoTECT v2 is a backwards-compatible extension of the
original PRoTECT (Wooding & Lavaei, ICCPS 2025) that adds several solver-side
improvements; this README documents what each improvement does and which
benchmarks rely on which features.

## How to run

```sh
cd submit
bash submit.sh
```

This builds the Docker image, runs `run_benchmarks.py` inside the container,
and extracts `results/results.csv` (plus per-benchmark JSON figures) onto the
host. `mosek.lic` (this folder) is copied into the image; if it is absent, the
script falls back to CVXOPT.

**`mosek.lic` auto-purge.** The license file is shipped inside the zip so the
portal's first Docker build can copy it into the image, but once the build is
complete it is no longer needed on disk. `submit.sh` installs an `EXIT/INT/TERM`
trap that deletes `mosek.lic` from `submit/` at the end of the run (or on
ctrl-C / error), so the file is gone after the script finishes regardless of
exit path. Re-running `submit.sh` after the file is purged will simply fall
back to CVXOPT, with the warning shown above.

Output schema in `results/results.csv`:

| column | meaning |
| --- | --- |
| `benchmark`, `instance` | benchmark id and per-family instance label |
| `result` | 1 = barrier found AND post-verification accepted, 0 = otherwise |
| `time` | per-benchmark wall time (seconds) |
| `b_degree`, `gamma`, `lambda` | barrier degree and level-set values |
| `solver` | `mosek` if MOSEK produced a barrier; `cvxopt` only if MOSEK was unable to |
| `sos_overall` | `clean` / `warning` / `fail` from the post-solve numerical validator |
| `barrier` | the SOS-form expression of the certificate |

## What's new in PRoTECT v2

Every benchmark in this submission uses the v2 features below. The original
PRoTECT (`ct_DS`) entry point is unchanged; the new code paths live in
`src/functions/`.

### 1. Post-solve numerical validation — `sos_validate.py` (used by **all** benchmarks)

Every barrier returned by MOSEK / CVXOPT is now validated post-hoc against the
solver's own SOS decomposition. PICOS surfaces three useful artefacts after a
solve: `prob.status`, `constraint.Qval` (the Gram matrix) and `constraint.b_sym`
(the monomial basis). In exact arithmetic, `b_sym^T · Q · b_sym` would equal
the asserted polynomial. In practice MOSEK's Q has two kinds of noise:

* tiny non-zero entries (~1e-10 to 1e-7) that should be zero, and
* coefficient drift on the genuinely non-zero entries proportional to the
  SDP feasibility tolerance (~1e-8 by default).

The default `SOSConstraint.get_sos_decomp(precision=3)` rounds the factored
polynomial to 3 decimal places after Cholesky, which amplifies the noise to
~1e-3 and makes residuals look much worse than the SDP's actual feasibility.

`sos_validate.cleaned_sos_decomposition` instead does a manual rounding-then-
PSD-projection pass on Q (round entries below 1e-8 to zero, symmetrise, clamp
negative eigenvalues to 0, then `V · sqrt(W)` Cholesky-style factorisation),
then substitutes the post-solve decision-variable values into the asserted
polynomial before computing the residual. This matches MOSEK's true precision
(~1e-8) and lets us distinguish "solver returned a clean certificate" from
"solver returned a numerically infeasible one".

`run_benchmarks.py` overrides `result=0` whenever `sos_overall == 'fail'`, so a
barrier is only reported as found if it survives validation.

### 2. Solver fallback — `solve_helpers.py` (used by **all** benchmarks)

`solve_safety_problem(degrees, x, f, ..., margin, mosek_tol)` sweeps the
degree list with MOSEK first; if MOSEK cannot produce a barrier at any
degree, it sweeps the same list with CVXOPT. The first MOSEK barrier wins
regardless of validation status (the residual is still recorded for the
report); CVXOPT is only consulted when MOSEK genuinely fails. This avoids
expensive retries on certificates that are numerically tight but acceptable.

### 3. Uncertain-parameter robustness — `ct_DS_robust.py` (used by all benchmarks; CVDP23/b_unc and TRAF22 exercise the parameter axis)

`ct_DS_robust` is a drop-in replacement for `ct_DS` that supports
**parameter-robust** barrier search: given dynamics `x' = f(x, p)` with
`p ∈ [P_lo, P_hi]`, it searches for `B(x)` (independent of `p`) such that
the unsafe set is unreachable from the initial set under **every** admissible
`p`. The Lie-derivative SOS constraint is encoded over `(x, p)` with a
fresh Positivstellensatz multiplier per parameter dimension:

```
g_param[k] = (p_k - P_lo[k]) * (P_hi[k] - p_k)   ≥ 0  iff  p_k ∈ [P_lo, P_hi]
```

The barrier basis stays in the n-dimensional state, so the SOS basis is
roughly `C(n+m+d, d) / C(n+d, d)` smaller than the lifting-based encoding
(treating `p` as an extra state with `p' = 0`). Every benchmark in this
suite uses `ct_DS_robust` even when the parameter list is empty, because
that gives us a uniform code path; benchmarks that actually use the
parameter axis are flagged below.

### 4. Decision-margin parameter (used by SPRE22 and LOVO21)

`ct_DS_robust(margin=...)` enforces `λ - γ ≥ margin` strictly, instead of
the original `λ > γ`. A non-zero margin gives the resulting certificate a
finite separation gap between the initial- and unsafe-side level sets, which
is necessary for tight numerical instances to validate cleanly under the
post-hoc check.

### 5. Tighter MOSEK tolerances (used by SPRE22 and LOVO21)

`ct_DS_robust(mosek_tol=...)` forwards `MSK_DPAR_INTPNT_CO_TOL_PFEAS`,
`MSK_DPAR_INTPNT_CO_TOL_DFEAS` and `MSK_DPAR_INTPNT_CO_TOL_REL_GAP` to MOSEK.
Default is `None` (MOSEK defaults, ~1e-8). Tight benchmarks set `1e-10`.

### 6. Sinc relaxation — `sinc_relaxation.py` (used by TRAF22 and TSPS25)

For benchmarks containing trigonometric dynamics, `sinc_relaxation` replaces
`sin(θ)` with `θ · σ(θ)` where `σ` is treated as an uncertain parameter
bounded in `[sinc(θ_max), 1]`. This converts the trig system into a
polynomial system in `(x, σ)` that `ct_DS_robust` can handle directly, with
the parameter-box S-procedure providing exact coverage of every admissible
`σ` value over the relaxation domain.

## Per-benchmark feature matrix

The two-digit suffix on each benchmark id is the ARCH-COMP NLN year it was
first proposed (e.g. `LALO20` was added in the 2020 round, `TSPS25` in 2025).
The "Year added" column makes that explicit; the "Submission status" column
distinguishes benchmarks that are **new for ARCH-COMP 2026** from
**carried-over benchmarks from previous years**.

| Benchmark | Year added | Submission status | Encoding | Robust params (`p_syms`) | Sinc relax | `margin` | `mosek_tol` | Post-validate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ROBE25/1,2,3 | 2025 | **new for 2026** (revised spec — autocatalytic Robertson) | `ct_DS_robust` | – | – | – | default | yes |
| CVDP23/b_unc | 2023 | **new for 2026** (newly uncertain `b ∈ [1, 3]` instance) | `ct_DS_robust` | `b ∈ [1, 3]` | – | – | default | yes |
| LOVO25 | 2025 | **new for 2026** (Lotka-Volterra) | `ct_DS_robust` | – | – | – | default | yes |
| TSPS25 | 2025 | **new for 2026** (3-state planar with trig dynamics) | `ct_DS_robust` | – | yes | – | default | yes |
| CVDP23/b2 | 2023 | carried over (Coupled van der Pol, `b=2` fixed) | `ct_DS_robust` | – | – | – | default | yes |
| CVDP22 | 2022 | carried over (Coupled van der Pol, `b=70`) | `ct_DS_robust` | – | – | – | default | yes |
| ROBE21/1,2,3 | 2021 | carried over (rescaled Robertson — original-spec fallback for ROBE25) | `ct_DS_robust` | – | – | – | default | yes |
| LOVO21 | 2021 | carried over (Lorenz — original-spec fallback for LOVO25) | `ct_DS_robust` | – | – | **4.0** | **1e-10** | yes |
| LALO20/W001 | 2020 | carried over (Lorenz, w=0.01) | `ct_DS_robust` | – | – | – | default | yes |
| LALO20/W005 | 2020 | carried over (Lorenz, w=0.05) | `ct_DS_robust` | – | – | – | default | yes |
| LALO20/W01 | 2020 | carried over (Lorenz, w=0.1) | `ct_DS_robust` | – | – | – | default | yes |
| SPRE22 | 2022 | carried over (4-D spread model) | `ct_DS_robust` | – | – | **10.0** | **1e-10** | yes |
| TRAF22 | 2022 | carried over (traffic flow) | `ct_DS_robust` | `σ ∈ [sinc_lo, 1]` | yes | – | default | yes |

## Spec versioning

Where the 2026 round revised an earlier spec, both the new-spec attempt and
the original-spec fallback are reported, so the write-up can show a "we
attempted the 2026 spec, fell back to the original" story:

* **New for 2026:** `ROBE25/*` (autocatalytic Robertson, revised spec); `CVDP23/b_unc` (Coupled van der Pol with uncertain `b`); `LOVO25` (Lotka-Volterra); `TSPS25` (3-state planar). PRoTECT v2's robust + sinc machinery was added specifically to attack the harder 2026 instances.
* **Carried over from previous years:** `ROBE21/*` (2021), `LOVO21` (2021), `CVDP22` (2022), `CVDP23/b2` (2023), `LALO20/{W001,W005,W01}` (2020), `SPRE22` (2022), `TRAF22` (2022). All are run with the same v2 pipeline (post-validation, robust encoding, MOSEK→CVXOPT fallback) so the comparison against the new instances is apples-to-apples.

## Layout

```
submit/
├── Dockerfile           # ubuntu:22.04 + python deps + PRoTECT v2 overlay
├── submit.sh            # build, run, docker cp results
├── mosek.lic            # MOSEK license (gitignored from the source repo)
├── results/             # populated by submit.sh on completion
└── data/
    ├── ex/ARCH-COMP/2026-NLN/
    │   ├── run_benchmarks.py   # benchmark runner (TIMEOUT=5000s)
    │   └── benchmarks/         # one script per (benchmark, instance) pair
    └── src/functions/
        ├── ct_DS_robust.py     # robust barrier search
        ├── solve_helpers.py    # MOSEK/CVXOPT fallback wrapper
        ├── sos_validate.py     # post-solve numerical validation
        ├── sinc_relaxation.py  # sin(θ) → θ·σ relaxation
        └── ... (other v2 helpers)
```
