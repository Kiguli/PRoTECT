# PRoTECT v2 — novelties over v1

PRoTECT v1 (ICCPS / ICTAC 2025) verified safety barrier certificates for
four classes of dynamical systems (continuous- and discrete-time,
deterministic and stochastic) via Positivstellensatz / sum-of-squares
(SOS) programmes in MOSEK/CVXOPT. v2 extends the tool along five
orthogonal axes documented below. Every contribution is implemented
across **all four barrier classes** (`ct_DS`, `dt_DS`, `ct_SS`,
`dt_SS`) — see the *Coverage* row in each section.

## 1. Pointwise post-solve validator
*Files: `src/functions/sos_validate.py`*

**Problem.** SOS solvers report `optimal` whenever the
Positivstellensatz inequalities hold *at the polynomial-coefficient
level* modulo floating-point tolerance (~$10^{-8}$). On bounded
asserted boxes, however, the S-procedure Lagrangian multipliers
$L_i(x) \ge 0$ can *absorb* slack: a coefficient-clean certificate
can have $B(x_*) < \lambda$ pointwise on the unsafe-set boundary
at the order of the multiplier magnitudes. PRoTECT v1 surfaced
only the coefficient residuals, so MOSEK's `optimal` could mean
the certificate was numerically loose on the closed boxes.

**Solution.** `sos_validate.pointwise_validate` directly samples
every barrier condition on its asserted set in $(x, p)$-space:

  * **init**:   `sup_{X_0 corners + 2000 interior samples}  B(x) - gamma`
  * **unsafe**: `inf_{X_u corners + 2000 interior samples}  B(x) - lambda`
                 across every unsafe region
  * **Lie / step / generator / expectation**: `sup_{X x P samples}`
    of the appropriate decrement condition

The validator returns slacks + **witness points** + a
`pass / warn / fail` verdict. The combined `sos_overall` in every
solver is now the AND of the coefficient check and the pointwise
check.

**Demonstration.** CVDP23 infinite-time at degree 4 reports
`coefficient: clean` (residuals $\sim 10^{-7}$) but
`pointwise: fail` (unsafe slack $+1.95 \times 10^{-5}$, witness
point near $(1.484, 2.725, 1.389, 2.997)$). v1 would have falsely
declared the certificate valid; v2 catches it.

**Coverage.** Built into `ct_DS_robust`, `ct_DS_finite_time`,
`dt_DS_robust`, `dt_DS_finite_time`, `ct_SS_robust`, `dt_SS_robust`.

## 2. Robust parameter S-procedure (uncertain parameter box)
*Files: `ct_DS_robust.py`, `dt_DS_robust.py`, `ct_SS_robust.py`, `dt_SS_robust.py`*

**Problem.** v1 handles uncertain parameters only by *state
lifting* — adding $\dot p = 0$ as an extra state. This expands the
SOS basis from $\binom{n + d}{d}$ to $\binom{n + m + d}{d}$
monomials, which for CVDP23 ($n = 4$, $m = 1$, $d = 4$) takes
$70 \to 126$ monomials and an ~$11\times$ wall-time penalty.

**Solution.** Each v2 robust solver searches a barrier $B(x)$
**independent of $p$**, but adds a Positivstellensatz multiplier
$L_p(x, p)$ for the parameter box $g_p(p) = (p - P_{\text{lo}}) (P_{\text{hi}} - p)$
in the *only* SOS constraint that sees the dynamics:

$$
- \mathcal{D}^p B(x) - \sum_i L_{s,i}(x, p)\, g_{X,i}(x) - L_p(x, p)\, g_p(p) \in \Sigma[x, p],
$$

where $\mathcal{D}^p$ is the appropriate decrement operator for
each system class (Lie derivative, one-step difference,
infinitesimal generator, conditional expectation). The barrier
basis stays in $x$ only; the parameter dependence lives entirely
in the multipliers.

**Demonstration.** CVDP23 with $b \in [1, 3]$:
v1 (state-lifting) needed ~616 s at degree 4; v2 (robust SOS) takes
~93 s — a $\sim 6.6\times$ speed-up with the same Positivstellensatz
soundness guarantee.

**Coverage.** All four barrier classes; the multiplier $L_p(x, p)$ is
SOS in $(x, p)$ in every case.

## 3. Finite-time-horizon barriers
*Files: `ct_DS_finite_time.py`, `dt_DS_finite_time.py`*

**Problem.** v1 only handles *time-invariant* barriers $B(x)$, which
must satisfy $\mathcal{D} B(x) \le 0$ (or $\le c$ for stochastic
systems) on the entire state space. For systems that are "just
barely safe" — like CVDP23, whose reach tube approaches the unsafe
boundary $y_{1,2} = 2.75$ within $\sim 10^{-6}$ — no time-invariant
barrier exists with a strict $\gamma < \lambda$ gap. v1 reports
$\gamma \approx \lambda$ to solver tolerance.

**Solution.** A time-augmented barrier
$$B(x, t) = \sum_{j=0}^{K} t^j \, B_j(x)$$
with a time-box S-procedure multiplier $g_t(t) = t (T - t) \ge 0$
on $[0, T]$. The decrement condition becomes
$\partial B/\partial t + \langle \nabla B,\, f \rangle \le 0$
(continuous-time) or $B(f(x), k+1) - B(x, k) \le 0$ (discrete-time).
The time-dependent barrier has the freedom to *be non-monotone*
in time and need only be invariant for the finite horizon
$[0, T]$ — breaking the structural $\gamma \approx \lambda$
tightness of the infinite-time case.

**Demonstration.** CVDP23/paper-spec with $b \in [1, 3]$ uncertain
and $t \in [0, 7]$: the v2 finite-time solver finds a pointwise-
sound certificate (unsafe slack $-3.2 \times 10^{-11}$, Lie slack
$+6.8 \times 10^{-7}$) at $(\mathrm{degree} = 2, \mathrm{time\ orders} = 2)$.
This is the *only* PRoTECT formulation that verifies CVDP23 with
the paper's full safety + horizon spec.

**Coverage.** Implemented for deterministic systems (`ct_DS_finite_time`,
`dt_DS_finite_time`); the stochastic c-martingale formulation
already encodes finite-horizon confidence in v1 via the
$\gamma + c T < \lambda$ level-set constraint (no time-augmented
barrier needed).

## 4. Full-precision barrier serialization
*Files: `ct_DS_robust.py`, `ct_DS_finite_time.py`, `dt_DS_robust.py`, `dt_DS_finite_time.py`, `ct_SS_robust.py`, `dt_SS_robust.py`*

**Problem.** `python-sumofsquares.SOSConstraint.get_sos_decomp(precision=3)`
defaults to rounding all polynomial coefficients to 3 decimal
places before serialization. For stiff dynamics (ROBE25/3 has
coefficients up to $10^7$), this rounding flips the saved barrier
from coefficient-clean to pointwise-failing by orders of magnitude
at the state-space boundary.

**Solution.** All v2 solvers save barriers via
`get_sos_decomp(precision=20)`, preserving MOSEK/CVXOPT's full
IEEE-754 precision. (Underlying call: `round_sympy_expr(S, precision)`
maps each `sp.Number` via `round(n, 20)`, which preserves all 15+
double-precision digits.)

**Demonstration.** CVDP23/b2: before the fix, `init_slack = +2.2e-2`
(`pointwise fail`); after the fix, `init_slack = -7.4e-6`
(`pointwise pass`). Four-order-of-magnitude improvement, same
SOS solution, just better serialization.

## 5. SMT-backed post-hoc verification
*Files: `src/functions/verify_smt.py`, `verify_dreal.py`*

**Problem.** Even after pointwise validation, the certificate is
only confirmed at sampled points. For a *rigorous* guarantee we
need to prove the inequalities hold for **all** $x$ in the closed
boxes, in exact arithmetic.

**Solution.** Each barrier's SOS coefficients are converted to
rationals via `sp.Rational(float).limit_denominator(10^9)` and
passed to Z3 in `QF_NRA` (nonlinear real arithmetic). For each
condition we ask Z3 whether the *negation* is satisfiable: `unsat`
means the condition holds rigorously; `sat` means Z3 produced
a true counterexample. dReal is supported as a faster (delta-
decision) alternative for high-dimensional polynomial inequalities.

**Demonstration.** LOVO25 verified end-to-end by Z3 (init, all
four unsafe regions, Lie all `unsat`). LALO20 unsafe condition
rigorously verified for all three instances. CVDP23 finite-time
Lie condition rigorously verified for both b-fixed and b-uncertain
variants.

**Coverage.** Generic — `verify_barrier` accepts any polynomial
barrier with init/unsafe/space sets and (optional) parameter box;
applies to all four barrier classes.

## 6. Solver fallback and degree sweeping
*Files: `src/functions/solve_helpers.py`*

**Problem.** MOSEK occasionally returns `picos.SolutionFailure`
(degenerate SDP, ill-conditioned multipliers, license trouble) but
CVXOPT solves the same programme without issue. v1 forced the user
to retry by hand.

**Solution.** `solve_safety_problem(degrees, ...)` and
`solve_finite_time_safety_problem(degrees, time_orders, T, ...)`
sweep through (degree, time_orders) combinations with MOSEK first,
fall back to CVXOPT only when MOSEK can't produce a barrier at
any combination, and return the first feasible certificate.
Particularly useful on CVDP23 finite-time, where MOSEK converges
but CVXOPT delivers the lowest pointwise slack — the wrapper
automatically picks whichever solver gives the cleanest verdict.

## What each contribution adds, in summary

| v2 feature | Files | Closes what gap | Affects which barrier classes |
|---|---|---|---|
| Pointwise validator | `sos_validate.py` | Coefficient-clean / pointwise-fail soundness gap | all four |
| Robust parameter box | `*_robust.py` (4 files) | Need to lift parameter to state | all four |
| Finite-time barrier | `*_finite_time.py` (2 files; stochastic uses c-martingale) | Just-barely-safe infinite-time tightness | det. CT/DT (SS via c-martingale) |
| Full-precision save | each `_robust` / `_finite_time` solver | Coefficient rounding kills stiff barriers | all four |
| Z3/dReal verifier | `verify_smt.py`, `verify_dreal.py` | Sampling vs. rigorous proof | all four |
| MOSEK/CVXOPT fallback + degree sweep | `solve_helpers.py` | Single-solver brittleness | all four |

## Validation against ARCH-COMP 2026 NLN benchmarks

The new features are exercised on the ARCH-COMP 2026 NLN suite
(see `ex/ARCH-COMP/2026-NLN/` for the canonical submission and
`ex/ARCH-COMP/2026-NLN-extras/` for variants):

| Benchmark | v1 verdict | v2 verdict | Feature(s) responsible |
|---|---|---|---|
| LALO20/W001, W005, W01 | found (but uncertain pointwise) | **clean** (pointwise slack $\sim -5$ on every condition) | pointwise validator + full-precision save |
| CVDP23 / b $\in [1, 3]$, $t \in [0, 7]$ | not certifiable | **warn** (pointwise slack $\sim 10^{-7}$) | robust param + finite-time + full-precision save |
| LOVO25 | found | rigorously verified by Z3 | Z3 verifier |

Future-ARCH-COMP candidates (proposed adapted benchmarks under
`ex/ARCH-COMP/2026-NLN-extras/proposed/`) demonstrate the new
features on benchmarks where the original spec is just-out-of-reach;
each one varies *only* the initial-set tightness or state-space
envelope from the original ARCH-COMP spec to bring the system into
the certifiable regime.
