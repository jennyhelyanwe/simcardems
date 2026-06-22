"""
src/simcardems/pseudo_ecg.py

Single-dipole pseudo-ECG, computed on-the-fly during the EP time loop
(not as post-processing), using the standard lead-field formula:

    phi_e(r) = (1 / (4 * pi * sigma_e)) * integral_Omega grad(V_m) . grad(1/|x - r|) dV

evaluated at each electrode position r, where V_m is the transmembrane
potential field already in memory each EP step. Constant prefactor
(sigma_e, 4*pi) is dropped/folded into a single scale constant since
absolute units matter less than relative waveform shape for most
pseudo-ECG use cases -- adjust `scale` if you need calibrated units.
"""

from typing import List, Tuple

import dolfin
import numpy as np

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl


class PseudoECGUpdater:
    """
    Precomputes one UFL form per electrode at setup time (since the
    1/|x - r| kernel only depends on fixed electrode positions, not on
    the solution), then assembles all 10 each step -- cheap, fully
    parallel (each assemble() is an MPI-collective integral over the
    mesh, scales with however the mesh itself is partitioned).
    """

    def __init__(
        self,
        mesh: dolfin.Mesh,
        v: dolfin.Function,
        electrode_positions: List[Tuple[float, float, float]],
        scale: float = 1.0,
        regularization: float = 1.0,
    ):
        """
        mesh: the EP mesh
        v: the transmembrane potential Function -- MUST be the actual
           solution Function cbcbeat updates in place each step (e.g.
           solver.solution_fields()[1].sub(0), or however vs/vur expose
           V_m -- NEEDS CONFIRMING against cbcbeat's actual field layout,
           see note below), not a copy, so grad(v) tracks the live solution.
        electrode_positions: list of 10 (x, y, z) tuples, mm, matching
                              the mesh's coordinate system
        scale: overall multiplicative constant (1 / (4*pi*sigma_e) in
               the textbook formula) -- left as a free parameter since
               sigma_e (bulk conductivity) is a modeling choice, not
               something we've derived; defaults to 1.0 (relative units)
        regularization: small constant added inside the distance term
                        to avoid a singularity if an electrode position
                        coincides exactly with a mesh point (1/|x-r|
                        blows up at x=r) -- only matters if electrodes
                        sit ON the mesh rather than outside it
        """
        self.mesh = mesh
        self.v = v
        self.electrode_positions = electrode_positions
        self.scale = scale

        x = dolfin.SpatialCoordinate(mesh)
        dx = dolfin.Measure("dx", domain=mesh)

        self.forms = []
        for r in electrode_positions:
            r_const = dolfin.Constant(r)
            dist = ufl.sqrt(ufl.dot(x - r_const, x - r_const) + regularization**2)
            kernel = 1.0 / dist
            integrand = ufl.dot(ufl.grad(v), ufl.grad(kernel))
            self.forms.append(integrand * dx)

        self.history = []  # list of (t, [10 values]) -- optional convenience buffer

    def compute(self) -> np.ndarray:
        """Assemble all 10 electrode integrals for the CURRENT solution state."""
        values = np.array([dolfin.assemble(form) * self.scale for form in self.forms])
        return values

    def update(self, t: float) -> np.ndarray:
        """Call once per EP step; also appends to self.history for convenience."""
        values = self.compute()
        self.history.append((t, values.copy()))
        return values

    def get_history_arrays(self):
        """Returns (times: (n,) array, values: (n, 10) array) from accumulated history."""
        if not self.history:
            return np.array([]), np.zeros((0, 10))
        times = np.array([h[0] for h in self.history])
        values = np.array([h[1] for h in self.history])
        return times, values