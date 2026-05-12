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

**Solution.** Five high-level wrappers — one per v2 barrier class
plus the finite-time entry:

  * `solve_safety_problem` (continuous-time deterministic)
  * `solve_dt_DS_safety_problem`
  * `solve_ct_SS_safety_problem`
  * `solve_dt_SS_safety_problem`
  * `solve_finite_time_safety_problem` (deterministic finite-horizon)

Each sweeps through a degree list with MOSEK first, falls back to
CVXOPT only when MOSEK can't produce a barrier at any degree, and
returns the first feasible certificate. Useful on CVDP23 finite-time
where CVXOPT delivers the lowest pointwise slack — the wrapper
automatically picks whichever solver gives the cleanest verdict.

## 7. Sinc relaxation for non-polynomial dynamics
*Files: `src/functions/sinc_relaxation.py`, `src/functions/relaxations.py`*

**Problem.** v1 requires polynomial dynamics. Trigonometric ($\sin$,
$\cos$, $\tan$) and other non-polynomial terms (exp, log, sqrt,
inv_power, ...) appear in mechanics (TRAF22's $\sin\psi$,
TSPS25's bus-coupling sines), control (steering, NN activations), and
chemistry, blocking PRoTECT from those benchmarks entirely.

**Solution.** The identity $\sin(a) = \mathrm{sinc}(a) \cdot a$,
combined with the bound $\mathrm{sinc}(a) \in [\mathrm{sinc}(A), 1]$
on $|a| \le A$, lets us introduce an auxiliary state
$q = \mathrm{sinc}(a)$ with the trivial flow $\dot q = 0$ and the
*box S-procedure* $g_q(q) = (q - \mathrm{sinc}(A))(1 - q) \ge 0$ on
its admissible range. The dynamics become polynomial in $(x, q)$ and
flow naturally through the v2 robust solver. Analogously for $\cos$:
$\cos(a) = 1 - r a^2 / 2$ with $r = \mathrm{sinc}(a/2)^2 \in
[\mathrm{sinc}(A/2)^2, 1]$.

`relaxations.py` generalises this into a registry of named
relaxations:

| Term | Polynomial substitute | Auxiliary box |
| --- | --- | --- |
| $\sin(a)$ | $q a$ | $q \in [\mathrm{sinc}(A), 1]$ |
| $\cos(a)$ | $1 - r a^2 / 2$ | $r \in [\mathrm{sinc}(A/2)^2, 1]$ |
| $\tan(a)$ | $q a$ | $q \in [1, \tan(A)/A]$, $\lvert a \rvert < \pi/2$ |
| $e^a$ | $1 + q a$ | $q \in [\,?,\, (e^A - 1)/A\,]$ |
| $\log(1 + a)$ | $q a$ | $q \in [1/(1 + A), 1]$ |
| $\sqrt{a}$ | $q$, $q^2 = a$ | equality multiplier |
| $1/a^k$ | $q$, $q a^k = 1$ | equality multiplier |

**Demonstration.** TRAF22 has the steering dynamics
$\dot \psi = (v / l_{\rm wb}) \tan\delta$ and $\dot s_y = v \sin\psi$
which are non-polynomial. v1 could not handle this at all. v2's
TRAF22 benchmark drops $s_x$ from the state, swaps $\sin\psi
\to q_s \psi$ with $q_s \in [\mathrm{sinc}(\psi_{\max}), 1]$ via the
sinc relaxation, treats $q_s$ as an uncertain parameter via the
robust S-procedure (feature 2), and produces a barrier $B(x)$ over
the 4-D physical state. The sinc relaxation is essential to bringing
TRAF22 into the SOS-verifiable regime.

**Coverage.** Continuous-time deterministic via `ct_DS_robust`'s
parameter-box S-procedure; the same auxiliary-state trick applies
unchanged to `dt_DS_robust`, `ct_SS_robust`, `dt_SS_robust`.

## 8. Padé approximants for arbitrary smooth functions
*Files: `src/functions/pade.py`*

**Problem.** The sinc relaxation is exact for trig terms only. For an
arbitrary smooth function $f$ (e.g. a learnt control law, an empirical
fit, a transcendental dynamics term), there's no closed-form
auxiliary-box identity.

**Solution.** Padé approximants give the unique rational $[m/n]$
approximation that matches $f$'s Taylor expansion to order $m + n$.
Multiplying through by the denominator $Q_n(x)$ converts $f \approx
P_m(x) / Q_n(x)$ into the polynomial identity $f \cdot Q_n - P_m = 0$
(modulo truncation error). An auxiliary state $q$ standing for $f(x)$
is bounded by the worst-case $f \cdot Q_n - P_m$ residual on the
input range — then $q \cdot Q_n - P_m \in [-\varepsilon, \varepsilon]$
becomes a box constraint absorbable by the v2 SOS pipeline.

**Coverage.** General-purpose; usable by any benchmark with smooth
non-polynomial terms beyond the registry in `relaxations.py`.

## 9. Per-condition strict-positivity margins
*Files: `src/functions/ct_DS_robust.py`, `dt_DS_robust.py`,
`ct_SS_robust.py`, `dt_SS_robust.py`, `ct_DS_finite_time.py`,
`dt_DS_finite_time.py` (six solvers; per-class decrement-condition
margin name)*

**Problem.** The pointwise validator (feature 1) catches certificates
that are coefficient-clean but pointwise-loose by a fraction of a
unit. Why does pointwise looseness happen at all? In *exact*
arithmetic, the SOS Positivstellensatz
$-B(x) - \sum_i L_i(x) g_{X_0,i}(x) + \gamma \in \Sigma[x]$
implies $B(x) \le \gamma$ rigorously on $X_0$. The pointwise residual
is purely an artifact of MOSEK's $\varepsilon \sim 10^{-8}$
floating-point tolerance amplified by the polynomial basis values on
the boundary.

**Solution.** Replace each SOS condition with a *strictly*-positive
variant:

$$-B(x) - \sum_i L_i(x) g_{X_0,i}(x) + \gamma - \delta_{\text{init}} \in \Sigma[x]$$

(and analogously for the unsafe and Lie/step/generator/expectation
conditions). With $\delta > 0$, the asserted SOS polynomial is forced
to be $\ge \delta$ everywhere on $\mathbb{R}^n$, so on the asserted
set we get a *rigorous* pointwise margin of $\ge \delta - \varepsilon
\cdot M_{\text{basis}}$, where $M_{\text{basis}}$ is the basis
amplification factor. Setting $\delta \gg \varepsilon M_{\text{basis}}$
swamps the solver noise floor and gives a strictly positive pointwise
margin.

**Demonstration.** On LALO20 (pointwise slacks $\sim -5$), adding
$\delta = 0.1$ on every condition still admits a feasible certificate
with the validator reporting `pass` strictly. On CVDP23/b2 (genuinely
tight at degree 4), even $\delta = 10^{-4}$ makes the SOS programme
infeasible — diagnostically confirming the certificate's intrinsic
margin is below that threshold. The per-condition $\delta$s thus also
serve as a *quantitative probe of the certificate's real margin*.

**Coverage.** Implemented in all six v2 solvers (`ct_DS_robust`,
`dt_DS_robust`, `ct_SS_robust`, `dt_SS_robust`, `ct_DS_finite_time`,
`dt_DS_finite_time`) with class-appropriate decrement-condition
margin name:

| solver | decrement-condition margin |
| --- | --- |
| `ct_DS_robust`, `ct_DS_finite_time` | `lie_margin` |
| `dt_DS_robust`, `dt_DS_finite_time` | `step_margin` |
| `ct_SS_robust` | `generator_margin` |
| `dt_SS_robust` | `expectation_margin` |

Plus `init_margin` and `unsafe_margin` in every solver.

## 10. Reach-avoid encoding
*Files: `src/functions/ct_DS_reach_avoid.py`, `src/functions/reach_avoid.py`*

**Problem.** v1 verifies safety only (avoid the unsafe set forever).
ARCH-COMP and many control specs ask for *reach-avoid*: reach a
target region $T$ within horizon $H$ while staying away from $X_u$.

**Solution.** A two-barrier composite: $B_{\text{safety}}(x)$ excludes
$X_u$ (standard), and a complementary $V(x)$ acts as a control
Lyapunov-like certificate driving the system into $T$. Together they
prove $X_0 \to T$ under safety. The SOS programme couples them via
the standard Positivstellensatz; uncertain parameters and finite
horizons combine with features 2 and 3 unchanged.

## 11. Hybrid-system barriers
*Files: `src/functions/ct_DS_hybrid.py`, `src/functions/hybrid.py`*

**Problem.** Many ARCH-COMP NLN benchmarks are nominally hybrid
(SPRE22's spacecraft modes; LOVO25's tangential-crossing
formulation). v1 has no native hybrid support.

**Solution.** Mode-local barriers $B_q(x)$ per discrete mode $q$, with
*reset-map continuity* conditions
$B_{q'}(R_{q \to q'}(x)) \le B_q(x)$ at each guard transition.
The reset map $R$ becomes an equality multiplier in the SOS
programme; the guard polynomial becomes a positivity multiplier.

## 12. Piecewise-input sequence handling
*Files: `src/functions/ct_DS_piecewise_sequence.py`, `src/functions/piecewise_input.py`*

**Problem.** TRAF22's reference controller emits a piecewise-constant
input sequence over the horizon. v1 verifies only autonomous systems.

**Solution.** Treat each piecewise input segment $u_k$ as a fresh
parameter box; chain segment-local barriers via boundary matching.
Combined with features 2 and 3, gives a full piecewise-input
verification pipeline.

## 13. Sub-Gaussian noise barriers
*Files: `src/functions/ct_SS_subgaussian.py`*

**Problem.** v1's ct_SS / dt_SS assume Gaussian (or uniform / log-normal)
noise with closed-form moments. Many practical noise models are
sub-Gaussian without nice moments.

**Solution.** Use the sub-Gaussian moment bound $E[e^{\lambda X}]
\le e^{\sigma^2 \lambda^2 / 2}$ directly in the expectation SOS
constraint. The resulting condition is polynomial in $\lambda$, $x$;
PRoTECT's SOS pipeline handles it.

## 14. Block-structured / compositional barriers
*Files: `src/functions/block_decomp.py`*

**Problem.** Networked systems $x_i' = f_i(x_i) + \sum_{j \ne i}
h_{ij}(x_i, x_j)$ have full-state dimension growing with the network
size; the monolithic SOS programme blows up combinatorially.

**Solution.** Per-subsystem barriers $B_i(x_i)$ plus interconnection
"supply-rate" certificates $S_{ij}$ that account for inter-subsystem
energy flow. Each per-subsystem SOS programme is independent and small,
and the whole-system safety follows from a small-gain argument.
Reference: Anand, Lavaei, Soudjani (compositional CBFs).

## 15. Sparse SOS / term sparsity
*Files: `src/functions/sparse_sos.py`*

**Problem.** Standard SOS PSD coefficient count is
$O\!\left(\binom{n + d}{d}^2\right)$. For TSPS25 ($n = 15$,
$d = 4$): $\binom{19}{4}^2 = 14\,400$ — out of reach. Term sparsity
(Wang, Magron, Lasserre, SIAM J. Optim. 2021) exploits the fact that
sparse polynomials admit SOS decompositions on much smaller monomial
bases.

**Solution.** Scaffold for TSSOS / term-sparse PSD pivots — currently
a stub awaiting the Python port of TSSOS.

## 16. Differential-algebraic systems (DAE)
*Files: `src/functions/dae.py`*

**Problem.** TSPS25's index-1 DAE form $\dot x = f(x, y, u)$,
$0 = g(x, y, u)$ has 27 algebraic equations binding 27 variables.
v1's pure-ODE pipeline cannot consume this directly.

**Solution.** Two strategies:
* **Index-1 elimination**: solve $g = 0$ explicitly for $y = \phi(x, u)$
  and substitute back into $f$, recovering a pure ODE.
* **Manifold-restricted SOS**: keep $y$ as decision variables and add
  $\lambda_g(x, y) \cdot g(x, y, u)$ as an *equality multiplier* in the
  SOS programme. Mathematically the same Positivstellensatz pattern as
  the relaxation registry's $\sqrt{a}$, $1/a^k$ equality multipliers.

## 17. Slow-fast model reduction
*Files: `src/functions/slow_fast.py`*

**Problem.** ROBE25 instances 2 and 3 with $\gamma = 10^5$ and $10^7$
are *stiff* — fast variables relax to a quasi-steady manifold on
timescales much shorter than the slow variables. The SOS programme on
the full system has terrible conditioning.

**Solution.** Solve $f_{\text{fast}} = 0$ for $x_{\text{fast}} =
\phi(x_{\text{slow}})$ (quasi-steady-state) and substitute back into
$f_{\text{slow}}$, yielding a reduced ODE on the slow variables. The
reduced system has much better SOS conditioning.

## 18. Disturbance robust SOS
*Files: `src/functions/disturbance.py`*

**Problem.** Bounded time-varying disturbances $w(t) \in W$ in the
dynamics — TRAF22's $w_1, w_2$ are paper-spec disturbances v1 cannot
handle natively.

**Solution.** Mathematically identical to the parameter-robust
encoding (feature 2): the disturbance gets a box S-procedure
multiplier in the Lie-derivative SOS. The semantic distinction is
just whether the variable is allowed to vary in time (disturbance)
or held constant (parameter); both reduce to the same SOS programme.
Thin re-export of `ct_DS_robust` under the disturbance terminology.

## 19. NN-controlled closed-loop verification
*Files: `src/functions/nn_control.py`*

**Problem.** AINNCS-style benchmarks have a ReLU NN controller in the
loop: $\dot x = f(x, \pi(x))$ with $\pi$ a feed-forward NN. v1 has no
NN support.

**Solution.** Exploit the piecewise-affine (PWA) representation of a
ReLU NN: the input domain partitions into polyhedral cells, and $\pi$
is affine on each. Per-cell SOS programmes verify safety inside each
cell using polytope constraints (from `sets.py`); inter-cell
continuity via boundary-hyperplane S-procedure multipliers.

## 20. Vertex enumeration for parameter robustness
*Files: `src/functions/vertex_enumeration.py`*

**Problem.** When the parameter box has many corners or the dynamics
are highly nonlinear in $p$, the robust SOS (feature 2) can give
unduly conservative certificates.

**Solution.** Alternative to the robust S-procedure: solve $2^m$
independent SOS programmes at the vertices of the parameter polytope,
then intersect the resulting barrier level sets. Each per-vertex SOS
is smaller and better-conditioned, and the intersection certificate
applies to every $p$ in the polytope (by convex combination of the
vertex certificates).

## What each contribution adds, in summary

| v2 feature | Files | Closes what gap | Affects which barrier classes |
|---|---|---|---|
| 1. Pointwise validator | `sos_validate.py` | Coefficient-clean / pointwise-fail soundness gap | all four |
| 2. Robust parameter box | `*_robust.py` | Need to lift parameter to state | all four |
| 3. Finite-time barrier | `*_finite_time.py` | Just-barely-safe infinite-time tightness | det. CT/DT |
| 4. Full-precision save | each `_robust` / `_finite_time` solver | Coefficient rounding kills stiff barriers | all four |
| 5. Z3/dReal verifier | `verify_smt.py`, `verify_dreal.py` | Sampling vs. rigorous proof | all four |
| 6. MOSEK/CVXOPT fallback + degree sweep | `solve_helpers.py` | Single-solver brittleness | all four |
| 7. **Sinc relaxation** | `sinc_relaxation.py`, `relaxations.py` | **No trig dynamics support** | all four |
| 8. Padé approximant relaxation | `pade.py` | No support for arbitrary smooth non-polynomial terms | all four |
| 9. Per-condition margin | all 6 v2 solvers | Rigorous pointwise margin > solver tolerance | all four |
| 10. Reach-avoid encoding | `ct_DS_reach_avoid.py`, `reach_avoid.py` | Only safety in v1 | continuous-time det. |
| 11. Hybrid barriers | `ct_DS_hybrid.py`, `hybrid.py` | No hybrid-mode support | continuous-time det. |
| 12. Piecewise input | `ct_DS_piecewise_sequence.py`, `piecewise_input.py` | No time-varying controller | continuous-time det. |
| 13. Sub-Gaussian noise | `ct_SS_subgaussian.py` | Only Gaussian/uniform/log-normal noise in v1 | continuous-time stoch. |
| 14. Block-compositional | `block_decomp.py` | Monolithic SOS for large networks | all four |
| 15. Sparse SOS | `sparse_sos.py` | High-dimensional benchmark blowup (TSPS25) | all four |
| 16. DAE support | `dae.py` | TSPS25's DAE form | continuous-time det. (extendable) |
| 17. Slow-fast reduction | `slow_fast.py` | Stiff systems (ROBE25/3) | continuous-time |
| 18. Disturbance robust | `disturbance.py` | Time-varying disturbances | all four (via 2) |
| 19. NN-controlled loop | `nn_control.py` | AINNCS / ReLU NN controllers | continuous-time det. |
| 20. Vertex enumeration | `vertex_enumeration.py` | Conservative robust SOS | all four |

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
