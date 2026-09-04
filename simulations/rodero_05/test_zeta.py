"""
test_zeta_formula.py — standalone check of the Land-model Zetas/Zetaw
ODE integration scheme, isolated from the full 3D simulation.

Real ODE: dZeta/dt = A*dLambda - c*Zeta  (dLambda held constant over one step)
True analytic solution:
    Zeta(t) = Zeta_prev*exp(-c*dt) + (A*dLambda/c)*(1 - exp(-c*dt))

As-written "analytic" scheme in active_model.py:
    Zeta_prev*exp(-c*dt) + (A*dLambda/c*dt)*(1 - exp(-c*dt))
    -- note the extra *dt on the second term, which the correct
    solution above does NOT have.
"""

import numpy as np

# Real parameter values, from TorLandFull.default_parameters()
As = Aw = 10.0
cs = 0.012 * 1.0 * 2.23 * 0.5 * (1 - 0.25) / 0.25   # ~0.04014
cw = 0.182 * 1.0 * 2.23 * (1 - 0.5) / 0.5             # ~0.40586


def zeta_as_written(Zeta_prev, A, c, dLambda, dt):
    return Zeta_prev * np.exp(-c * dt) + (A * dLambda / c * dt) * (1.0 - np.exp(-c * dt))


def zeta_corrected(Zeta_prev, A, c, dLambda, dt):
    return Zeta_prev * np.exp(-c * dt) + (A * dLambda / c) * (1.0 - np.exp(-c * dt))


def zeta_ground_truth(Zeta_prev, A, c, dLambda, dt, n_substeps=10000):
    """Fine-grained forward Euler — trusted reference, independent of
    either analytic formula above."""
    Zeta = Zeta_prev
    sub_dt = dt / n_substeps
    for _ in range(n_substeps):
        Zeta += sub_dt * (A * dLambda - c * Zeta)
    return Zeta


print(f"{'quantity':6s} {'dt':>8s} {'dLambda':>10s} {'as-written':>14s} {'corrected':>14s} {'ground truth':>14s}")
for quantity, A, c in [("Zetas", As, cs), ("Zetaw", Aw, cw)]:
    for dt in [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        for dLambda in [0.001, 0.01, 0.05]:
            Zeta_prev = 0.0  # start from rest, single step — isolates the formula itself
            v_written = zeta_as_written(Zeta_prev, A, c, dLambda, dt)
            v_corrected = zeta_corrected(Zeta_prev, A, c, dLambda, dt)
            v_truth = zeta_ground_truth(Zeta_prev, A, c, dLambda, dt)
            print(f"{quantity:6s} {dt:8.2f} {dLambda:10.4f} {v_written:14.6f} {v_corrected:14.6f} {v_truth:14.6f}")