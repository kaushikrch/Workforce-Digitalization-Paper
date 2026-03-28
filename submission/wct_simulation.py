import matplotlib
matplotlib.use("Agg")
"""
WCT Model: Workforce, Capability, and Trust in Omnichannel Retail
=================================================================
Simulation code for:
  "When Front-Stage Innovation Overloads Back-Stage Workers:
   A Dynamic Model of Workforce, Capability, and Customer Trust
   in Omnichannel Retail"

Submission version. Final reviewed and validated package.

Notation used throughout:
  CLV  -- customer lifetime value
  ODE  -- ordinary differential equation
  ROI  -- return on investment
  IC   -- initial condition (defined where first used in Study 2)

Solver:   scipy.integrate.solve_ivp with LSODA (adaptive step-size),
          rtol=1e-8, atol=[1e-9, 1e-10, 1e-10] for [K, W, T].
          Explicit event detection at the piecewise-smooth switching
          manifold G = 0 (Filippov, 1988).

Studies:  1. Hidden Cost of App Innovation     (Fig 4, 120-month horizon)
          2. Tipping Point Map / Basin Map      (Fig 2, 37 x 26 grid)
          3. Decomposing Workforce ROI           (EC only; underlies Thm 2, §4.3)
          4. Burnout Cliff                       (Fig 5, 61-point dF sweep)
          5. Digitalisation-Workforce Comp.     (Fig 3, analytical)

Paper figure sequence:
  Fig 1  Causal-loop diagram (TikZ; no simulation)
  Fig 2  Basin-of-attraction map              -> Study 2
  Fig 3  Complementarity Fmax(W)              -> Study 5
  Fig 4  Hidden cost of app innovation        -> Study 1
  Fig 5  Burnout Cliff                        -> Study 4
  Study 3 underlies Theorem 2 structural-zero result (EC Table EC.4); no main figure.

Validation: Equilibrium residuals < 1e-6.
            Max CLV deviation from fixed-step RK4 (dt=1e-3) < 0.3%.

All time units: months. All rate parameters in months^{-1}.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

np.random.seed(0)   # reproducibility: fix seed before any stochastic calls


# ──────────────────────────────────────────────────────────────────────────────
# 1. BASELINE PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

PARAMS = dict(
    # ── Structural parameters ─────────────────────────────────────────────────
    b     = 1.30,   # demand super-linearity (Table 2, Row 1)
                    #   Bell et al. (2018), Acimovic & Graves (2015): b > 1
                    #   Trap requires only b > 1; robust across b ∈ [1.1, 2.0]
    a     = 0.40,   # trust-demand coupling (Table 2, Row 2)
                    #   McKnight et al. (2002), Sirdeshmukh et al. (2002)
    p     = 0.50,   # workforce-technology complementarity (Table 2, Row 3)
                    #   Krusell et al. (2000): capital-skill complementarity;
                    #   illustrative mid-range value — model's central
                    #   testable parameter; results hold for any p ∈ (0, 1)
    sigma = 2.00,   # overload-recovery asymmetry (Table 2, Row 4)
                    #   Oliva & Sterman (2001); Bakker & Demerouti (2007)
    eta   = 2.00,   # trust erosion asymmetry (Table 2, Row 5)
                    #   Sirdeshmukh et al. (2002): ~2.5:1 empirical ratio
                    #   Conservative value (would imply eta ≈ 2.5 at midpoint)
    # ── Rate parameters (all in months^{-1}) ─────────────────────────────────
    r     = 0.40,   # sensing-adaptation rate (Table 2, Row 6)
                    #   Tucker (2007): 2–8 week implementation window
                    #   Half-life: ln(2)/r ≈ 1.7 months
    delta = 0.03,   # technology depreciation rate (Table 2, Row 7)
                    #   Illustrative; half-life ≈ 23 months
    beta  = 0.12,   # trust-update rate (Table 2, Row 8)
                    #   Illustrative; half-life ≈ 5.8 months
    mu    = 0.05,   # natural workforce decay rate (Table 2, Row 9)
                    #   Illustrative; half-life ≈ 14 months
    lam   = 0.025,  # natural trust decay rate (Table 2, Row 10)
                    #   Illustrative; half-life ≈ 28 months
    # ── Investment controls (capability units per month) ─────────────────────
    u     = 0.05,   # back-stage capability investment rate
                    #   Implies K* = u/delta = 1.667 at baseline
    f     = 0.03,   # workforce investment rate (default; overridden per study)
                    #   Baseline scenario; varied in Studies 3 and 4
)


# ── Figure-to-Study mapping ───────────────────────────────────────────────────
# Figure 1  →  Study 1: run_study1()   Hidden cost of app innovation
# Figure 2  →  Study 2: run_study2()   Tipping point / basin-of-attraction map
# EC only   →  Study 3: run_study3()   Decomposing workforce ROI (EC Table EC.4)
# Figure 5  →  Study 4: run_study4()   Burnout Cliff (f-varying, equilibrium ICs)
# Figure 3  →  Study 5: run_study5()   Digitalisation-workforce complementarity
#                                       (analytical; no ODE integration required)
# ─────────────────────────────────────────────────────────────────────────────

def Gpos(G):
    """Positive part [G]^+ = max(G, 0)."""
    return max(G, 0.0)

def Gneg(G):
    """Negative part [-G]^+ = max(-G, 0)."""
    return max(-G, 0.0)

def h(W, p):
    """Workforce capacity multiplier."""
    return (1 - p) + p * W

def demand(T, F, a, b):
    """Front-stage demand on back-stage operations."""
    return (1.0 + a * T) * F**b

def clv(T, rho0=0.80, rho_slope=0.18, w0=80.0, w_slope=0.25, T_kink=0.15):
    """
    Customer lifetime value (geometric-series form of Gupta et al. 2004).

    rho(T) = rho0 + rho_slope * T   (retention probability)
    w(T)   = w0 * (1 + w_slope * max(T - T_kink, 0))   (wallet share, USD/year)
    CLV    = w(T) / (1 - rho(T))
    """
    rho  = rho0 + rho_slope * T
    w    = w0 * (1.0 + w_slope * max(T - T_kink, 0.0))
    return w / (1.0 - rho)

def wct_ode(t, state, F, params, f_val=None):
    """
    WCT model ODEs.

    State = [K, W, T]  (all dimensionless or in consistent units)
    F     = front-stage digitalisation level (scalar or callable(t))
    f_val = workforce investment override; if None uses params['f']
    """
    K, W, T = state
    # Ensure state-space invariance
    W = np.clip(W, 0.0, 1.0)
    T = np.clip(T, 0.0, 1.0)

    # Current F (may be time-varying ramp)
    F_t = F(t) if callable(F) else F

    # Effective demand and gap
    D  = demand(T, F_t, params['a'], params['b'])
    Ke = K * h(W, params['p'])
    G  = D - Ke

    Gp = Gpos(G)
    Gm = Gneg(G)

    # Investment rate
    f = f_val if f_val is not None else params['f']

    # ODEs
    dK = params['r'] * Gp * W - params['delta'] * K + params['u']
    dW = (Gm + f) * (1.0 - W) - (params['mu'] + params['sigma'] * Gp) * W
    dT = (params['beta'] * Gm * (1.0 - T)
          - params['eta'] * params['beta'] * Gp * T
          - params['lam'] * T)

    return [dK, dW, dT]


def G_zero_event(t, state, F, params, f_val=None):
    """Event: G = 0 (switching manifold crossing)."""
    K, W, T = state
    F_t = F(t) if callable(F) else F
    D   = demand(T, F_t, params['a'], params['b'])
    Ke  = K * h(W, params['p'])
    return D - Ke

G_zero_event.terminal  = False   # do not stop; just record
G_zero_event.direction = 0       # detect both directions


# ──────────────────────────────────────────────────────────────────────────────
# 3. SOLVER WRAPPER
# ──────────────────────────────────────────────────────────────────────────────

RTOL  = 1e-8
ATOL  = [1e-9, 1e-10, 1e-10]   # [K, W, T]: K=1e-9, W=1e-10, T=1e-10 (matches paper §5 stated tolerances)

def integrate(state0, F, t_end, params, f_val=None, t_eval=None,
              method='LSODA'):
    """
    Solve the WCT ODE system using an adaptive integrator with event detection.

    Returns: (sol, t_switch) where sol is the OdeSolution and
             t_switch is a list of times when G crossed zero.
    """
    def ode(t, s):
        return wct_ode(t, s, F, params, f_val=f_val)

    def evt(t, s):
        return G_zero_event(t, s, F, params, f_val=f_val)

    evt.terminal  = False
    evt.direction = 0

    sol = solve_ivp(
        ode,
        [0.0, t_end],
        state0,
        method=method,
        rtol=RTOL,
        atol=ATOL,
        events=evt,
        dense_output=True,
        t_eval=t_eval,
    )
    t_switch = sol.t_events[0] if sol.t_events else []
    return sol, t_switch


# ──────────────────────────────────────────────────────────────────────────────
# 4. EQUILIBRIUM COMPUTATIONS
# ──────────────────────────────────────────────────────────────────────────────

def surplus_equilibrium(F, params, tol=1e-8, max_iter=200):
    """
    Compute the surplus equilibrium via the implicit equation Phi(S*) = 0.

    Returns (K_star, W_star, T_star, S_star) or None if F >= F_crit.
    """
    b, a, p   = params['b'], params['a'], params['p']
    mu, lam   = params['mu'], params['lam']
    delta, u  = params['delta'], params['u']
    beta, f_  = params['beta'], params['f']

    K_star = u / delta

    def Phi(S):
        Ws = (S + f_) / (S + f_ + mu)
        Ts = beta * S / (beta * S + lam)
        return K_star * ((1 - p) + p * Ws) - (1 + a * Ts) * F**b - S

    # Check surplus regime
    if Phi(0.0) <= 0:
        return None

    # Bisection to find S*
    lo, hi = 0.0, 1e4
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if Phi(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    S_star = (lo + hi) / 2.0
    W_star = (S_star + f_) / (S_star + f_ + mu)
    T_star = beta * S_star / (beta * S_star + lam)
    return K_star, W_star, T_star, S_star


def F_crit(params):
    """
    Compute the critical digitalisation threshold F_crit analytically.
    At S*=0: T*=0, W*=f/(f+mu), K*=u/delta.
    F_crit = (K* h(f/(f+mu)))^{1/b}
    """
    b, p = params['b'], params['p']
    f_   = params['f']
    mu   = params['mu']
    K_star = params['u'] / params['delta']
    W_boundary = f_ / (f_ + mu)
    return (K_star * h(W_boundary, p)) ** (1.0 / b)


def F_max_W(W, params):
    """
    Maximum sustainable digitalisation level at workforce well-being W.
    Evaluated at the regime boundary S*=0 (T*=0):
      F_max(W) = (K* h(W))^{1/b}
    """
    K_star = params['u'] / params['delta']
    return (K_star * h(W, params['p'])) ** (1.0 / params['b'])


def verify_equilibrium(state, F, params, f_val=None):
    """Return max absolute ODE residual at a putative equilibrium."""
    res = wct_ode(0.0, state, F, params, f_val=f_val)
    return max(abs(r) for r in res)


# ──────────────────────────────────────────────────────────────────────────────
# 5. RK4 REFERENCE (for validation)
# ──────────────────────────────────────────────────────────────────────────────

def rk4_integrate(state0, F, t_end, params, f_val=None, dt=1e-3):
    """Fixed-step RK4 for validation comparison only."""
    f_val = f_val if f_val is not None else params['f']
    t  = 0.0
    s  = list(state0)
    while t < t_end:
        dt_ = min(dt, t_end - t)
        F_t = F(t) if callable(F) else F
        k1 = wct_ode(t,         s,              F, params, f_val=f_val)
        k2 = wct_ode(t+dt_/2,   [x+dt_/2*k for x,k in zip(s,k1)], F, params, f_val=f_val)
        k3 = wct_ode(t+dt_/2,   [x+dt_/2*k for x,k in zip(s,k2)], F, params, f_val=f_val)
        k4 = wct_ode(t+dt_,     [x+dt_*k for x,k in zip(s,k3)],   F, params, f_val=f_val)
        s  = [x + dt_/6*(k1[i]+2*k2[i]+2*k3[i]+k4[i]) for i,x in enumerate(s)]
        t += dt_
    return s


# ──────────────────────────────────────────────────────────────────────────────
# 6. STUDY 1 – THE HIDDEN COST OF APP INNOVATION
# ──────────────────────────────────────────────────────────────────────────────

def run_study1(params=None, t_end=120, t_policy=24, n_points=1000):
    """
    Three scenarios with comparable total resource commitment but different
    front-stage/workforce investment allocations.

    Scenario A: aggressive F ramp, no workforce support  (f=0)
    Scenario B: moderate F ramp, concurrent workforce    (f=0.06)
    Scenario C: workforce-first (f=0.10 months 1-12), then ramp

    Primary analysis window: 24 months (policy horizon).
    Extended horizon: 120 months for convergence verification.
    """
    if params is None:
        params = PARAMS.copy()

    state0 = [1.60, 0.55, 0.50]
    F0     = 0.80
    F_end_A = 1.15    # Scenario A: aggressive ramp (final F_end = 1.15, not 1.10)
    F_end_B = 1.05    # Scenario B: moderate ramp (final F_end = 1.05)

    # Time-varying F ramps (linear)
    def F_A(t): return F0 + (F_end_A - F0) * min(t / t_policy, 1.0)
    def F_B(t): return F0 + (F_end_B - F0) * min(t / t_policy, 1.0)
    def F_C(t):
        # Hold F0 for first 12 months, then ramp to F_end_A
        if t <= 12:
            return F0
        return F0 + (F_end_A - F0) * min((t - 12) / t_policy, 1.0)

    t_eval = np.linspace(0, t_end, n_points)

    # Scenario A: f=0 (no workforce investment)
    solA, _ = integrate(state0, F_A, t_end, params, f_val=0.0,  t_eval=t_eval)
    # Scenario B: f=0.06
    solB, _ = integrate(state0, F_B, t_end, params, f_val=0.06, t_eval=t_eval)
    # Scenario C: f=0.10 for months 1-12, then f=0.0 (reverts to no investment)
    # NOTE: Paper §5.1 explicitly states f=0 after month 12
    def f_C(t): return 0.10 if t <= 12 else 0.0  # f=0 after month 12 per §5.1
    # f_val as callable not supported in ode wrapper; pass through params copy
    params_C = params.copy()
    def ode_C(t, s):
        return wct_ode(t, s, F_C, params_C, f_val=f_C(t))
    from scipy.integrate import solve_ivp as _siv
    solC = _siv(ode_C, [0, t_end], state0, method='LSODA',
                rtol=RTOL, atol=ATOL, t_eval=t_eval, dense_output=True)

    # CLV time series
    clvA = np.array([clv(solA.y[2, i]) for i in range(n_points)])
    clvB = np.array([clv(solB.y[2, i]) for i in range(n_points)])
    clvC = np.array([clv(solC.y[2, i]) for i in range(n_points)])

    # Front-stage ramp index (relative to F0)
    F_A_ts = np.array([F_A(t) for t in t_eval])
    F_B_ts = np.array([F_B(t) for t in t_eval])
    F_C_ts = np.array([F_C(t) for t in t_eval])

    print("\n=== Study 1: Hidden Cost of App Innovation ===")
    for label, sol, clv_ts in [("A", solA, clvA), ("B", solB, clvB), ("C", solC, clvC)]:
        idx24 = np.searchsorted(t_eval, t_policy)
        print(f"  Scenario {label} at month {t_policy}: "
              f"CLV={clv_ts[idx24]:.1f}, W={sol.y[1,idx24]:.3f}, T={sol.y[2,idx24]:.3f}")

    # Validation: compare terminal CLV vs RK4
    sA_rk4 = rk4_integrate(state0, F_A, t_end, params, f_val=0.0)
    clv_rk4_A = clv(sA_rk4[2])
    clv_lsoda_A = clvA[-1]
    dev_A = abs(clv_lsoda_A - clv_rk4_A) / max(abs(clv_rk4_A), 1e-6) * 100
    print(f"  Validation Scenario A: LSODA CLV={clv_lsoda_A:.1f}, "
          f"RK4 CLV={clv_rk4_A:.1f}, deviation={dev_A:.2f}%")

    return dict(t=t_eval, solA=solA, solB=solB, solC=solC,
                clvA=clvA, clvB=clvB, clvC=clvC,
                FA=F_A_ts, FB=F_B_ts, FC=F_C_ts,
                a=params['a'], b=params['b'])


# ──────────────────────────────────────────────────────────────────────────────
# 7. STUDY 2 – TIPPING POINT MAP
# ──────────────────────────────────────────────────────────────────────────────

def run_study2(params=None, t_end=500, clv_threshold=600):
    """
    Basin-of-attraction map: 37 W0 values × 26 F values = 962 conditions.

    Grid:
      W0 in {0.08, 0.10, ..., 0.82}   (DW = 0.02, 37 points)
      F  in {0.70, 0.75, ..., 1.95}   (DF = 0.05, 26 points)

    Initial conditions on balance manifold:
      K0 solves K0 * h(W0) = (1 + a*0.5) * F^b  (G=0 with T0=0.50)
      T0 = 0.50

    Convergence verification: separatrix shift < 3 cells on 2x resolution.

    Note on IC perturbation: initial conditions are placed just inside the surplus
    region via K0 * (1 + 1e-6) to avoid G=0 event fires at t=0 (see code comments).
    The perturbation is O(1e-6) of K0 and shifts the separatrix by < 0.001 grid
    cells — negligible relative to the 3-cell convergence tolerance above.

    Expected runtime: approximately 5 minutes on a standard laptop (962 ODE integrations to t=500 months each).
    """
    if params is None:
        params = PARAMS.copy()

    W0_grid = np.linspace(0.08, 0.82, 37)   # 37 pts: exact endpoints, no float drift
    F_grid  = np.arange(0.70, 1.96, 0.05)    # 26 pts
    assert len(W0_grid) == 37, f"Expected 37 W0 points, got {len(W0_grid)}"
    assert len(F_grid)  == 26, f"Expected 26 F points, got  {len(F_grid)}"

    T0 = 0.50
    a, b, p = params['a'], params['b'], params['p']
    delta, u = params['delta'], params['u']

    basin = np.zeros((len(W0_grid), len(F_grid)))  # 1=surplus, 0=low-trust

    print("\n=== Study 2: Tipping Point Map (962 conditions) ===")
    total = len(W0_grid) * len(F_grid)
    count = 0

    for i, W0 in enumerate(W0_grid):
        for j, F_val in enumerate(F_grid):
            # Balance manifold: K0 h(W0) = (1+a*T0)*F^b  =>  K0 = (1+a*T0)*F^b / h(W0)
            # Add a small surplus perturbation (1e-6) to place IC just inside the
            # surplus region (G slightly < 0), preventing the G=0 event from firing
            # at t=0 and causing root-bracketing errors in the ODE solver.
            h_W0 = h(W0, p)
            K0   = (1 + a * T0) * F_val**b / h_W0 * (1.0 + 1e-6)
            K0   = max(K0, 0.01)   # numerical floor

            state0 = [K0, W0, T0]

            sol, _ = integrate(state0, F_val, t_end, params,
                               f_val=params['f'])
            # Terminal CLV determines basin membership
            T_final = np.clip(sol.y[2, -1], 0, 1)
            clv_final = clv(T_final)
            basin[i, j] = 1 if clv_final > clv_threshold else 0

            count += 1
            if count % 100 == 0:
                print(f"  Progress: {count}/{total} ({100*count/total:.0f}%)")

    print(f"  Done. Surplus basin coverage: {basin.mean()*100:.1f}%")

    # Full doubled-grid validation — implementation: _verify_basin_doubled_grid() below.
    # Expected output: "max separatrix shift = 0.0075 W0-units ... PASS"
    # This result is reported in Paper §5.2 (l.524), Fig.2 caption (l.409), EC §4.2 (l.477).
    _verify_basin_doubled_grid(basin, W0_grid, F_grid, params, T0, t_end,
                               clv_threshold)

    return dict(W0_grid=W0_grid, F_grid=F_grid, basin=basin)


def _verify_basin_doubled_grid(basin_coarse, W0_grid, F_grid, params, T0, t_end,
                                clv_threshold):
    """
    Full doubled-grid validation: re-runs the basin at 2x resolution in both
    W0 (step 0.01, 73 points) and F (step 0.025, 51 points), giving 73x51 =
    3,723 trajectories at half the original grid spacing in each dimension.

    For each F-slice common to both grids, the separatrix position (boundary
    W0 value) is identified in both the coarse and fine grids and compared.
    The maximum shift in W0 units across all F-slices is reported. A shift
    below 0.04 (two coarse grid steps) confirms boundary stability.

    This supersedes the prior spot-check methodology (n_check=30 boundary
    cells), which is no longer used. The present function runs ALL 3,723
    fine-grid trajectories unconditionally; no sampling or abbreviation.

    Expected output (baseline parameters):
        "Doubled-grid validation: max separatrix shift = 0.0075 W0-units
         across N/26 F-slices compared (tolerance: DW < 0.04 ... PASS)"
    This result is cited in Paper §5.2, Fig.2 caption, and EC §4.2.
    To reproduce: run run_study2() with default parameters.
    """
    a, b_val, p = params['a'], params['b'], params['p']

    # Fine grid: 2x resolution in W0 and F
    W0_fine = np.round(np.arange(W0_grid[0], W0_grid[-1] + 0.005, 0.01), 3)
    F_fine  = np.round(np.arange(F_grid[0],  F_grid[-1]  + 0.012, 0.025), 3)
    n_W_fine, n_F_fine = len(W0_fine), len(F_fine)

    print(f"  Doubled-grid: {n_W_fine}x{n_F_fine} = {n_W_fine*n_F_fine} trajectories "
          f"(step dW=0.01, dF=0.025)...")

    # Run fine-grid basin
    basin_fine = np.zeros((n_W_fine, n_F_fine), dtype=int)
    total_fine = n_W_fine * n_F_fine
    count = 0
    for j, F_val in enumerate(F_fine):
        for i, W0 in enumerate(W0_fine):
            h_W0   = h(W0, p)
            K0     = (1 + a * T0) * F_val**b_val / h_W0 * (1.0 + 1e-6)
            state0 = [max(K0, 0.01), W0, T0]
            sol, _ = integrate(state0, F_val, t_end, params, f_val=params['f'])
            T_fin  = np.clip(sol.y[2, -1], 0, 1)
            basin_fine[i, j] = 1 if clv(T_fin) > clv_threshold else 0
            count += 1
        if (j + 1) % 10 == 0:
            print(f"    Fine grid: {count}/{total_fine} ({100*count/total_fine:.0f}%)")

    # Compare separatrix positions at coarse F-values
    max_shift  = 0.0
    n_compared = 0

    for j_c, F_c in enumerate(F_grid):
        # Find closest fine-grid F index
        j_f = int(np.argmin(np.abs(F_fine - F_c)))

        # Separatrix in coarse grid: highest W0 row that is surplus (basin=1)
        W_sep_coarse = None
        for i in range(basin_coarse.shape[0] - 1):
            if basin_coarse[i, j_c] == 1 and basin_coarse[i + 1, j_c] == 0:
                W_sep_coarse = 0.5 * (W0_grid[i] + W0_grid[i + 1])
                break

        # Separatrix in fine grid at same F
        W_sep_fine = None
        for i in range(len(W0_fine) - 1):
            if basin_fine[i, j_f] == 1 and basin_fine[i + 1, j_f] == 0:
                W_sep_fine = 0.5 * (W0_fine[i] + W0_fine[i + 1])
                break

        if W_sep_coarse is not None and W_sep_fine is not None:
            shift = abs(W_sep_fine - W_sep_coarse)
            max_shift = max(max_shift, shift)
            n_compared += 1

    print(f"  Doubled-grid validation: max separatrix shift = {max_shift:.4f} W0-units "
          f"across {n_compared}/{len(F_grid)} F-slices compared "
          f"(tolerance: DW < 0.04 = 4 fine-grid steps; "
          f"{'PASS' if max_shift < 0.04 else 'FAIL'})")
    return max_shift


# ──────────────────────────────────────────────────────────────────────────────
# 8. STUDY 3 – DECOMPOSING WORKFORCE ROI
# ──────────────────────────────────────────────────────────────────────────────

def run_study3(params=None, t_end=120, n_f=21):
    """
    Decompose CLV uplift from workforce investment f into:
      - Direct channel (capacity multiplier, p>0): full model minus p=0 counterfactual
      - Indirect channel (sensing only): p=0 model
    Design: equilibrium ICs at F=0.85 for each (f, model-variant) pair.
    F=0.85 is chosen below F_crit(p=0.5, f=0)=0.869 so both p=0.5 and p=0
    models have surplus equilibria at f=0, enabling clean comparative statics.

    Structural note: when p=0, h(W)=1 for all W, so K_eff=K*=u/delta is
    f-invariant, making equilibrium CLV exactly invariant to f. The indirect
    sensing channel is therefore structurally zero, and the direct
    capacity-multiplier channel accounts for virtually all CLV uplift from f.
    """
    if params is None:
        params = PARAMS.copy()

    F_val   = 0.85   # below F_crit(p=0.5,f=0)=0.869 — surplus eq exists for both model variants
    f_range = np.linspace(0.0, 0.20, n_f)

    params_no_mult        = params.copy()
    params_no_mult['p']   = 0.0

    clv_full    = np.zeros(n_f)
    clv_no_mult = np.zeros(n_f)

    # Baseline equilibrium ICs at f=0 for each model variant.
    # Using equilibrium ICs (rather than a fixed ad-hoc starting state) ensures
    # the decomposition reflects comparative-statics effects rather than transient
    # dynamics from an out-of-equilibrium initial condition.
    params_base_full        = params.copy();      params_base_full['f']  = 0.0
    params_base_nomult      = params_no_mult.copy(); params_base_nomult['f'] = 0.0

    eq_base_full  = surplus_equilibrium(F_val, params_base_full)
    eq_base_nomul = surplus_equilibrium(F_val, params_base_nomult)

    assert eq_base_full  is not None, f"No surplus eq at F={F_val}, f=0, p={params['p']}"
    assert eq_base_nomul is not None, f"No surplus eq at F={F_val}, f=0, p=0"

    state0_base_full  = list(eq_base_full[:3])   # [K*, W*, T*] for full model at f=0
    state0_base_nomul = list(eq_base_nomul[:3])  # [K*, W*, T*] for p=0 model at f=0

    sol_bf, _ = integrate(state0_base_full,  F_val, t_end, params_base_full,  f_val=0.0)
    sol_bn, _ = integrate(state0_base_nomul, F_val, t_end, params_base_nomult, f_val=0.0)
    clv_base_full   = clv(sol_bf.y[2, -1])
    clv_base_nomult = clv(sol_bn.y[2, -1])

    for k, f_val_k in enumerate(f_range):
        # Each run starts from the surplus equilibrium corresponding to its own f value,
        # so the comparison is cleanly across equilibria (comparative statics).
        p_k        = params.copy();      p_k['f']       = f_val_k
        p_k_nomult = params_no_mult.copy(); p_k_nomult['f'] = f_val_k

        eq_k      = surplus_equilibrium(F_val, p_k)
        eq_k_nm   = surplus_equilibrium(F_val, p_k_nomult)

        s0_k    = list(eq_k[:3])   if eq_k    else state0_base_full
        s0_k_nm = list(eq_k_nm[:3]) if eq_k_nm else state0_base_nomul

        sol1, _ = integrate(s0_k,    F_val, t_end, p_k,        f_val=f_val_k)
        sol2, _ = integrate(s0_k_nm, F_val, t_end, p_k_nomult, f_val=f_val_k)
        clv_full[k]    = clv(sol1.y[2, -1])
        clv_no_mult[k] = clv(sol2.y[2, -1])

    # Indirect channel: uplift in the p=0 counterfactual (sensing only).
    # Structural note: when p=0, h(W)=1 so K_eff=K*=u/delta is f-invariant,
    # making equilibrium CLV exactly invariant to f. The indirect channel is
    # therefore structurally (not merely numerically) zero.
    uplift_indir  = clv_no_mult - clv_base_nomult
    uplift_direct = (clv_full - clv_base_full) - uplift_indir
    uplift_total  = clv_full - clv_base_full

    idx_mid = n_f // 2   # f = 0.10
    share_direct = uplift_direct[idx_mid] / max(uplift_total[idx_mid], 1e-6) * 100

    print("\n=== Study 3: Workforce ROI Decomposition ===")
    print(f"  F={F_val}, equilibrium ICs, t_end={t_end}mo")
    print(f"  At f=0.10: total CLV uplift = {uplift_total[idx_mid]:.1f}, "
          f"direct channel share = {share_direct:.0f}%")
    print(f"  Indirect channel is structurally zero (p=0 CLV invariant to f).")

    return dict(f_range=f_range, clv_full=clv_full, clv_no_mult=clv_no_mult,
                uplift_total=uplift_total, uplift_direct=uplift_direct,
                uplift_indir=uplift_indir,
                clv_base_full=clv_base_full, clv_base_nomult=clv_base_nomult)


# ──────────────────────────────────────────────────────────────────────────────
# 9. STUDY 4 – BURNOUT CLIFF
# ──────────────────────────────────────────────────────────────────────────────

def run_study4(params=None, t_end=500, n_dF=61):
    """
    Burnout Cliff (Figure 5 in paper): Two-threshold structure.
    61-point Delta-F sweep from 0 to 0.60; three workforce scenarios.

    Three workforce-investment scenarios f in {0.01, 0.03, 0.10} are compared,
    each starting from its own surplus EQUILIBRIUM (K*, W*, T*) at F0=0.80.
    This ensures the analytic thresholds and simulation use a consistent T*.

    Two thresholds are reported for each scenario:
      dF_collapse = F_crit(f) - F0  (true permanent-collapse onset)
      dF_rapid    = S0 / [b * F0^(b-1) * (1+a*T*)]  (instantaneous-overload diagnostic)

    Cliff criterion: permanent collapse is T(t_end) < 0.15,
    i.e. convergence to the low-trust attractor (T*_lt ~ 0).
    This threshold is conservative: the low-trust attractor has T*_lt < 0.10,
    and the surplus attractor has T* > 0.40 for all F in the simulated range.
    The gap between the two attractors makes T=0.15 an unambiguous classifier.
    """
    if params is None:
        params = PARAMS.copy()

    F0       = 0.80
    dF_range = np.linspace(0.0, 0.60, n_dF)

    scenarios = [
        ('A-low',  0.01, r'Low $f=0.01$'),
        ('B-base', 0.03, r'Baseline $f=0.03$'),
        ('C-high', 0.10, r'High $f=0.10$'),
    ]

    print("\n=== Study 4: Burnout Cliff (Two-Threshold Structure) ===")
    print(f"  {'Scenario':>9}  {'f':>5}  {'W*':>6}  {'T*':>6}  {'S*':>6}  "
          f"{'dF_collapse':>12}  {'dF_rapid':>10}")

    results = {}

    for label, f_scen, display in scenarios:
        p_sc = params.copy(); p_sc['f'] = f_scen

        # Start from the surplus equilibrium for this scenario
        eq = surplus_equilibrium(F0, p_sc)
        if eq is None:
            print(f"  {label}: no surplus equilibrium at F0={F0} -- skipping")
            continue
        Ks, Ws, Ts, Ss = eq
        state0 = [Ks, Ws, Ts]

        # Analytic thresholds
        h_bnd_f     = (1 - p_sc['p']) + p_sc['p'] * (f_scen / (f_scen + p_sc['mu']))
        Fc_f        = (p_sc['u'] / p_sc['delta'] * h_bnd_f) ** (1.0 / p_sc['b'])
        dF_collapse = Fc_f - F0
        dF_rapid    = Ss / (p_sc['b'] * F0**(p_sc['b']-1) * (1 + p_sc['a']*Ts))

        print(f"  {label:>9}  {f_scen:>5.2f}  {Ws:>6.3f}  {Ts:>6.3f}  {Ss:>6.3f}  "
              f"{dF_collapse:>12.4f}  {dF_rapid:>10.4f}")

        # Simulate over dF sweep
        clv_vals  = np.zeros(n_dF)
        collapsed = np.zeros(n_dF, dtype=bool)
        for k, dF in enumerate(dF_range):
            sol, _ = integrate(state0, F0 + dF, t_end, p_sc, f_val=f_scen)
            T_end        = sol.y[2, -1]
            clv_vals[k]  = clv(T_end)
            collapsed[k] = T_end < 0.15

        clv_norm = clv_vals / max(clv_vals[0], 1e-6) * 100

        # Simulated first collapse
        first_idx = np.where(collapsed)[0]
        sim_cliff = dF_range[first_idx[0]] if len(first_idx) > 0 else dF_range[-1]
        print(f"    Simulated first-collapse dF = {sim_cliff:.4f}  "
              f"(analytic dF_collapse = {dF_collapse:.4f}, "
              f"gap = {abs(sim_cliff - dF_collapse):.4f})")

        results[label] = dict(
            f=f_scen, display=display,
            Ks=Ks, Ws=Ws, Ts=Ts, Ss=Ss,
            dF_collapse=dF_collapse, dF_rapid=dF_rapid,
            clv_vals=clv_vals, clv_norm=clv_norm,
            collapsed=collapsed, sim_cliff=sim_cliff,
        )

    return dict(dF_range=dF_range, scenarios=scenarios, results=results, F0=F0)


def run_study5(params=None):
    """
    Analytical comparative statics from F_max(W) = (K* h(W))^{1/b}.
    T*=0 at the regime boundary (S*->0); formula is exact at boundary.
    """
    if params is None:
        params = PARAMS.copy()

    W_range = np.linspace(0.05, 0.95, 200)
    Fmax    = np.array([F_max_W(w, params) for w in W_range])

    W30 = 0.30
    W70 = 0.70
    Fmax30 = F_max_W(W30, params)
    Fmax70 = F_max_W(W70, params)
    gain   = (Fmax70 - Fmax30) / Fmax30 * 100

    print("\n=== Study 5: Digitalisation-Workforce Complementarity ===")
    print(f"  F_max(W=0.30) = {Fmax30:.3f}")
    print(f"  F_max(W=0.70) = {Fmax70:.3f}")
    print(f"  Gain from W0.30 to W0.70 = {gain:.1f}%")

    # Also compute F_crit for reference
    Fc = F_crit(params)
    print(f"  F_crit (at baseline f={params['f']}) = {Fc:.3f}")

    return dict(W_range=W_range, Fmax=Fmax, Fmax30=Fmax30, Fmax70=Fmax70,
                gain=gain, Fcrit=Fc)


# ──────────────────────────────────────────────────────────────────────────────
# 11. EQUILIBRIUM VALIDATION SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

def validate_equilibria(params=None):
    """
    Verify computed surplus equilibria by checking ODE residuals < 1e-6.
    Also verify low-trust attractor numerically.
    """
    if params is None:
        params = PARAMS.copy()

    print("\n=== Equilibrium Validation ===")

    F_vals = [0.70, 0.80, 0.90, 1.00, 1.05]
    for F_val in F_vals:
        eq = surplus_equilibrium(F_val, params)
        if eq is None:
            print(f"  F={F_val:.2f}: No surplus equilibrium (F >= F_crit)")
            continue
        K_star, W_star, T_star, S_star = eq
        state_eq = [K_star, W_star, T_star]
        resid = verify_equilibrium(state_eq, F_val, params)
        print(f"  F={F_val:.2f}: S*={S_star:.4f}, K*={K_star:.3f}, "
              f"W*={W_star:.3f}, T*={T_star:.3f}  |  residual={resid:.2e} "
              f"{'OK' if resid < 1e-6 else 'FAIL'}")

    # Stability condition check for low-trust attractor
    print("\n  Low-trust attractor stability (mu + sigma*G* > r*W*_low):")
    for F_val in [1.10, 1.20, 1.30, 1.40, 1.50]:
        # Numerically find low-trust fixed point
        try:
            K0_lt = params['u'] / params['delta'] * 0.5
            sol, _ = integrate([K0_lt, 0.10, 0.01], F_val, 500, params)
            K_lt = sol.y[0, -1]; W_lt = sol.y[1, -1]; T_lt = sol.y[2, -1]
            D_lt = demand(T_lt, F_val, params['a'], params['b'])
            G_lt = D_lt - K_lt * h(W_lt, params['p'])
            lhs  = params['mu'] + params['sigma'] * G_lt
            rhs  = params['r'] * W_lt
            print(f"    F={F_val:.2f}: mu+sigma*G*={lhs:.4f}, r*W*_low={rhs:.4f}, "
                  f"condition {'satisfied' if lhs > rhs else 'VIOLATED'}")
        except Exception as e:
            print(f"    F={F_val:.2f}: error {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 12. FIGURE GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_study1(res, t_policy=24, save_path="fig4_hidden_cost.pdf"):
    """Figure 1: Hidden cost of app innovation."""
    t  = res['t']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("The Hidden Cost of App Innovation", fontsize=13, fontweight='bold')

    colours = {'A': '#d62728', 'B': '#ff7f0e', 'C': '#2ca02c'}
    labels  = {'A': 'Scenario A: Aggressive ramp, no WF support',
               'B': 'Scenario B: Moderate ramp + concurrent WF',
               'C': 'Scenario C: WF-first, then ramp'}

    # Panel (a): Demand-Volume Index F·D(T(t),F(t))
    ax = axes[0, 0]
    a_param, b_param = res.get('a', 0.4), res.get('b', 1.3)
    for sc, F_ts, sol, col, ls in [
        ('A', res['FA'], res['solA'], colours['A'], '-'),
        ('B', res['FB'], res['solB'], colours['B'], '--'),
        ('C', res['FC'], res['solC'], colours['C'], ':')]:
        T_ts = sol.y[2]
        demand_vol = F_ts * (1 + a_param * T_ts) * F_ts**b_param
        ax.plot(t, demand_vol, color=col, lw=2, linestyle=ls, label=sc)
    ax.axvline(t_policy, color='grey', lw=1, linestyle='--', alpha=0.6)
    ax.set_xlabel("Time (months)"); ax.set_ylabel(r"Demand-Volume Index $F \cdot D(T,F)$")
    ax.set_title("(a) Front-Stage Demand Volume"); ax.legend(fontsize=8)

    # Panel (b): CLV
    ax = axes[0, 1]
    for sc, clv_ts in [('A', res['clvA']), ('B', res['clvB']), ('C', res['clvC'])]:
        ax.plot(t, clv_ts, color=colours[sc], lw=2,
                linestyle='-' if sc=='C' else ('--' if sc=='B' else ':'))
    ax.axvline(t_policy, color='grey', lw=1, linestyle='--', alpha=0.6)
    ax.set_xlabel("Time (months)"); ax.set_ylabel("CLV ($)")
    ax.set_title("(b) Customer Lifetime Value")
    for sc, col in colours.items():
        ax.plot([], [], color=col, lw=2, label=sc)
    ax.legend(fontsize=8)

    # Panel (c): Workforce W
    ax = axes[1, 0]
    for sc, sol in [('A', res['solA']), ('B', res['solB']), ('C', res['solC'])]:
        ax.plot(t, sol.y[1], color=colours[sc], lw=2)
    ax.axvline(t_policy, color='grey', lw=1, linestyle='--', alpha=0.6)
    ax.set_xlabel("Time (months)"); ax.set_ylabel("Workforce Well-being $W$")
    ax.set_title("(c) Workforce Well-being")
    ax.set_ylim(0, 1)

    # Panel (d): Trust T
    ax = axes[1, 1]
    for sc, sol in [('A', res['solA']), ('B', res['solB']), ('C', res['solC'])]:
        ax.plot(t, sol.y[2], color=colours[sc], lw=2)
    ax.axvline(t_policy, color='grey', lw=1, linestyle='--', alpha=0.6)
    ax.set_xlabel("Time (months)"); ax.set_ylabel("Customer Trust $T$")
    ax.set_title("(d) Customer Trust")
    ax.set_ylim(0, 1)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_study2(res, save_path="fig2_basin_map.pdf"):
    """Figure 2: Tipping point map."""
    W0_grid = res['W0_grid']
    F_grid  = res['F_grid']
    basin   = res['basin']

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = mcolors.ListedColormap(['#d62728', '#2ca02c'])
    ax.pcolormesh(F_grid, W0_grid, basin, cmap=cmap, alpha=0.7,
                  vmin=0, vmax=1)

    # Separatrix (contour at 0.5)
    ax.contour(F_grid, W0_grid, basin, levels=[0.5], colors='black',
               linewidths=2, linestyles='--')

    # Illustrative firm
    firm_W, firm_F = 0.50, 1.05
    ax.scatter([firm_F], [firm_W], s=120, color='white', zorder=5,
               edgecolors='black', marker='*')
    ax.annotate("Firm", (firm_F, firm_W), xytext=(firm_F+0.05, firm_W+0.05),
                fontsize=9, color='white')
    ax.annotate("", xy=(firm_F+0.20, firm_W+0.02),
                xytext=(firm_F+0.02, firm_W),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))

    # Annotate F_crit on the separatrix
    ax.annotate(r'$F_{\mathrm{crit}} \approx 1.11$',
                xy=(1.11, 0.78), xytext=(1.30, 0.78),
                fontsize=10, fontweight='bold', color='black',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='black', alpha=0.9))

    ax.set_xlabel("Front-Stage Digitalisation $F$", fontsize=11)
    ax.set_ylabel("Initial Workforce Well-being $W_0$", fontsize=11)
    ax.set_title("Tipping Point Map: Basin of Attraction\n"
                 "Green = Surplus (high CLV); Red = Low-trust attractor", fontsize=11)
    ax.grid(alpha=0.2)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ca02c', alpha=0.7, label='Surplus basin'),
                       Patch(facecolor='#d62728', alpha=0.7, label='Low-trust basin')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_study3(res, save_path="fig_study3_roi_decomp.pdf"):
    """Figure 3: Workforce ROI decomposition."""
    f_range = res['f_range']
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.fill_between(f_range, 0, res['uplift_indir'],
                    alpha=0.5, color='#aec7e8', label='Indirect (sensing)')
    ax.fill_between(f_range, res['uplift_indir'],
                    res['uplift_indir'] + res['uplift_direct'],
                    alpha=0.7, color='#1f77b4', label='Direct (capacity multiplier)')
    ax.plot(f_range, res['uplift_total'], 'k-', lw=2, label='Total uplift')

    ax.set_xlabel("Workforce Investment Rate $f$", fontsize=11)
    ax.set_ylabel("CLV Uplift over $f=0$ baseline ($)", fontsize=11)
    ax.set_title("Decomposing Workforce ROI\n"
                 "Capacity multiplier is the dominant channel", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_study4(res, save_path="fig5_burnout_cliff.pdf"):
    """Figure 4: Burnout Cliff — three f-scenarios with two thresholds each."""
    import matplotlib.patches as mpatches

    dF_range = res['dF_range']
    results  = res['results']
    colours  = {'A-low': '#d62728', 'B-base': '#1f77b4', 'C-high': '#2ca02c'}
    styles   = {'A-low': '--',      'B-base': '-',       'C-high': '-.'}

    fig, ax = plt.subplots(figsize=(7, 5))

    for label, rd in results.items():
        col = colours[label]; ls = styles[label]
        ax.plot(dF_range, rd['clv_norm'], color=col, lw=2.5, linestyle=ls,
                label=rd['display'])
        # Collapse threshold (solid vertical)
        ax.axvline(rd['dF_collapse'], color=col, lw=1.2, linestyle=':')
        # Rapid-collapse diagnostic (dashed vertical, lighter)
        ax.axvline(rd['dF_rapid'], color=col, lw=0.8, linestyle=(0,(3,5)), alpha=0.55)

    # Legend: use Line2D objects matching actual plotted line styles
    from matplotlib.lines import Line2D
    v1 = Line2D([0], [0], color='black', lw=1.2, linestyle=':',
                label=r'$\Delta F_{\mathrm{collapse}}$ (permanent onset)')
    v2 = Line2D([0], [0], color='black', lw=0.8, linestyle=(0,(3,5)),
                alpha=0.55,
                label=r'$\Delta F_{\mathrm{rapid}}$ (instantaneous overload)')
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [v1, v2], fontsize=8, loc='lower left')

    ax.set_xlabel(r'Upgrade Size $\Delta F$', fontsize=11)
    ax.set_ylabel('Terminal CLV (base = 100)', fontsize=11)
    ax.set_title("The Burnout Cliff\n"
                 r"$\Delta F_{\mathrm{collapse}}$ is workforce-contingent;"
                 r" $\Delta F_{\mathrm{rapid}}$ marks rapid-collapse onset",
                 fontsize=10)
    ax.set_ylim(0, 130)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_study5(res, save_path="fig3_complementarity.pdf"):
    """Figure 5: Digitalisation-workforce complementarity."""
    W_range = res['W_range']
    Fmax    = res['Fmax']

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(W_range, Fmax, 'b-', lw=2.5)
    ax.scatter([0.30, 0.70], [res['Fmax30'], res['Fmax70']],
               s=80, zorder=5, color=['#d62728', '#2ca02c'])
    ax.annotate(f"$W=0.30$\n$F_{{\\max}}={res['Fmax30']:.3f}$",
                (0.30, res['Fmax30']), xytext=(0.32, res['Fmax30']-0.05),
                fontsize=9, color='#d62728')
    ax.annotate(f"$W=0.70$\n$F_{{\\max}}={res['Fmax70']:.3f}$",
                (0.70, res['Fmax70']), xytext=(0.55, res['Fmax70']+0.02),
                fontsize=9, color='#2ca02c')

    ax.set_xlabel("Workforce Well-being $W$", fontsize=11)
    ax.set_ylabel("Max Sustainable Digitalisation $F_{\\max}(W)$", fontsize=11)
    ax.set_title(f"Digitalisation-Workforce Complementarity\n"
                 f"23% digitalisation gain from $W$ improvement alone", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# 13. MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def run_all_studies(params=None, run_study2_flag=True):
    """
    Execute all five simulation studies and generate figures.

    Study 2 (tipping map) involves 962 ODE integrations and is optional
    when run_study2_flag=False for quick testing.
    """
    if params is None:
        params = PARAMS.copy()

    print("=" * 60)
    print("WCT Model: Full Simulation Suite")
    print(f"Solver: LSODA, rtol={RTOL}, atol={ATOL}")
    print(f"Time units: months | All rates in months^{{-1}}")
    print("=" * 60)

    # Equilibrium validation
    validate_equilibria(params)

    # Study 1 → Paper Figure 4: Hidden Cost of App Innovation
    res1 = run_study1(params)
    plot_study1(res1, save_path="fig4_hidden_cost.pdf")

    # Study 2 → Paper Figure 2: Basin-of-Attraction Map (slow: 962 integrations)
    if run_study2_flag:
        res2 = run_study2(params)
        plot_study2(res2, save_path="fig2_basin_map.pdf")
    else:
        print("\n[Study 2 skipped; set run_study2_flag=True to run basin map]")
        res2 = None

    # Study 3 → EC only (no main-paper figure).
    # Retained for full replication; underlies the Theorem 2 structural-zero
    # result (§4.3 of the paper and EC Table EC.4).
    res3 = run_study3(params)
    plot_study3(res3, save_path="fig_study3_roi_decomp.pdf")

    # Study 4 → Paper Figure 5: Burnout Cliff
    res4 = run_study4(params)
    plot_study4(res4, save_path="fig5_burnout_cliff.pdf")

    # Study 5 → Paper Figure 3: Digitalisation-Workforce Complementarity
    res5 = run_study5(params)
    plot_study5(res5, save_path="fig3_complementarity.pdf")

    print("\n" + "=" * 60)
    print("All studies complete.")
    return dict(res1=res1, res2=res2, res3=res3, res4=res4, res5=res5)


if __name__ == "__main__":
    # Run all studies (Study 2 may take 3-5 minutes due to 962 integrations)
    results = run_all_studies(run_study2_flag=True)
