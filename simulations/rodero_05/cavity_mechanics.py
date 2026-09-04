"""
Cavity-volume-constrained (Lagrange multiplier) mechanics for IVC/IVR,
and the dual-cavity mode manager that switches between volume- and
pressure-driven boundary conditions per cavity.
"""

import numpy as np
import dolfin
import pulse
from pulse import kinematics as _kinematics
from pulse.dolfin_utils import list_sum as _list_sum
import pulse.mechanicsproblem as _mp
try:
    import ufl_legacy as _ufl
except ImportError:
    import ufl as _ufl

from simcardems import utils

logger = utils.getLogger(__name__)


# ── Monkey-patch: normal-only Robin BC (frictionless contact) ───────────
def _external_work_normal_robin(self, u, v):
    F = dolfin.variable(_kinematics.DeformationGradient(u))
    N = self.geometry.facet_normal
    ds = self.geometry.ds
    dx = self.geometry.dx
    external_work = []
    for neumann in self.bcs.neumann:
        n = neumann.traction * _ufl.cofac(F) * N
        external_work.append(dolfin.inner(v, n) * ds(neumann.marker))
    for robin in self.bcs.robin:
        u_normal = dolfin.inner(u, N) * N
        r = robin.value * u_normal
        external_work.append(dolfin.inner(r, v) * ds(robin.marker))
    for body_force in self.bcs.body_force:
        external_work.append(-dolfin.derivative(dolfin.inner(body_force, u) * dx, u, v))
    return _list_sum(external_work) if external_work else None


_mp.MechanicsProblem._external_work = _external_work_normal_robin
logger.info("Patched MechanicsProblem._external_work: Robin BC normal-only")


class CavityConstrainedMechanicsProblem(pulse.MechanicsProblem):
    def __init__(self, *args, cavity_specs=None, **kwargs):
        if not cavity_specs:
            raise ValueError("cavity_specs must have at least one {marker: v_target}")
        self.cavity_specs = cavity_specs
        self._u_standalone = None
        self._u_assigner = None
        super().__init__(*args, **kwargs)

    def _init_spaces(self):
        mesh = self.geometry.mesh
        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        R_elements = [dolfin.FiniteElement("Real", mesh.ufl_cell(), 0) for _ in self.cavity_specs]
        self.state_space = dolfin.FunctionSpace(mesh, dolfin.MixedElement([P2, P1] + R_elements))
        self.state = dolfin.Function(self.state_space, name="state")
        self.state_test = dolfin.TestFunction(self.state_space)

    def _init_forms(self):
        parts = dolfin.split(self.state)
        test_parts = dolfin.split(self.state_test)
        u, p = parts[0], parts[1]
        v, q = test_parts[0], test_parts[1]
        P_cavities = parts[2:]
        markers = list(self.cavity_specs.keys())

        F = dolfin.variable(_kinematics.DeformationGradient(u))
        J = _kinematics.Jacobian(F)
        mesh = self.geometry.mesh
        dx = self.geometry.dx
        ds = self.geometry.ds
        N = self.geometry.facet_normal

        internal_energy = self.material.strain_energy(F) + self.material.compressibility(p, J)
        self._virtual_work = dolfin.derivative(internal_energy * dx, self.state, self.state_test)

        if self.strong_coupling:
            f0 = self.material.active.f0
            f = F * f0
            valve_mask = getattr(self.geometry, "valve_mask", None)
            if valve_mask is not None:
                myocardium = 1.0 - valve_mask
                Pa_frozen = myocardium * self.material.active.Ta_current * dolfin.outer(f, f0)
            else:
                Pa_frozen = self.material.active.Ta_current * dolfin.outer(f, f0)
            self._virtual_work += dolfin.inner(Pa_frozen, dolfin.grad(v)) * dx

        external_work = self._external_work(u, v)
        if external_work is not None:
            self._virtual_work += external_work

        x = dolfin.SpatialCoordinate(mesh) + u
        n_weighted = _ufl.cofac(F) * N
        cavity_integrand = -dolfin.dot(x, n_weighted) / 3.0
        total_ref_volume = dolfin.assemble(dolfin.Constant(1.0) * dx)

        cavity_energy = 0
        for marker, P_c in zip(markers, P_cavities):
            v_target = self.cavity_specs[marker]
            cavity_energy += -P_c * cavity_integrand * ds(marker)
            cavity_energy += (P_c * v_target / total_ref_volume) * dx
        self._virtual_work += dolfin.derivative(cavity_energy, self.state, self.state_test)

        self._dirichlet_bc = []
        self._set_dirichlet_bc()
        self._jacobian = dolfin.derivative(self._virtual_work, self.state, dolfin.TrialFunction(self.state_space))
        self._init_solver()

    @property
    def strong_coupling(self):
        return hasattr(self.material.active, "Ta")

    def solve(self):
        nliter, nlconv = self._raw_solve()
        self._update_active_stress_bookkeeping()
        return nliter, nlconv

    def _raw_solve(self):
        return super().solve()

    def _update_active_stress_bookkeeping(self, update_ta_current=True):
        if not self.strong_coupling:
            return
        active = self.material.active

        u = get_u_view(self)
        F = dolfin.grad(u) + dolfin.Identity(3)
        f = F * active.f0
        lmbda = dolfin.sqrt(f ** 2)
        active._projector.project(active.lmbda, lmbda)
        if active.dt > 0:
            active._projector.project(active._dLambda, (lmbda - active.lmbda_prev) / active.dt)

        if update_ta_current:
            active._projector.project(active.Ta_current, active.Ta(lmbda))
            ta_arr = active.Ta_current.vector().get_local()
            ta_arr[ta_arr < 0.0] = 0.0  # projection-ringing clamp
            valve_mask = getattr(self.geometry, "valve_mask", None)
            if valve_mask is not None:
                mask_arr = valve_mask.vector().get_local()
                ta_arr[mask_arr > 0.5] = 0.0
            active.Ta_current.vector().set_local(ta_arr)
            active.Ta_current.vector().apply("insert")

        active.update_current(lmbda=lmbda)
        active.update_prev()

    def get_cavity_pressure(self, marker):
        idx = list(self.cavity_specs.keys()).index(marker)
        sub_idx = 2 + idx
        if not hasattr(self, '_p_standalone'):
            self._p_standalone, self._p_assigner = {}, {}
        if idx not in self._p_standalone:
            V_full = self.state_space
            R = V_full.ufl_element().sub_elements()[sub_idx]
            V_r = dolfin.FunctionSpace(V_full.mesh(), R)
            self._p_standalone[idx] = dolfin.Function(V_r)
            self._p_assigner[idx] = dolfin.FunctionAssigner(V_r, V_full.sub(sub_idx))
        self._p_assigner[idx].assign(self._p_standalone[idx], self.state.sub(sub_idx))
        local = self._p_standalone[idx].vector().get_local()
        local_val = float(local[0]) if len(local) > 0 else 0.0
        return dolfin.MPI.sum(self.geometry.mesh.mpi_comm(), local_val)


def get_u_view(problem):
    if getattr(problem, '_u_standalone', None) is None:
        V_full = problem.state_space
        P2 = V_full.ufl_element().sub_elements()[0]
        V_u = dolfin.FunctionSpace(V_full.mesh(), P2)
        problem._u_standalone = dolfin.Function(V_u)
        problem._u_assigner = dolfin.FunctionAssigner(V_u, V_full.sub(0))
    problem._u_assigner.assign(problem._u_standalone, problem.state.sub(0))
    return problem._u_standalone

def get_p_view(problem):
    """Split-free extraction of the P1 incompressibility pressure field
    (state.sub(1)) — same FunctionAssigner pattern as get_u_view."""
    if getattr(problem, '_p_field_standalone', None) is None:
        V_full = problem.state_space
        P1 = V_full.ufl_element().sub_elements()[1]
        V_p = dolfin.FunctionSpace(V_full.mesh(), P1)
        problem._p_field_standalone = dolfin.Function(V_p)
        problem._p_field_assigner = dolfin.FunctionAssigner(V_p, V_full.sub(1))
    problem._p_field_assigner.assign(problem._p_field_standalone, problem.state.sub(1))
    return problem._p_field_standalone


def handoff_uP(source_problem, dest_problem):
    assigner_u = dolfin.FunctionAssigner(dest_problem.state_space.sub(0), source_problem.state_space.sub(0))
    assigner_u.assign(dest_problem.state.sub(0), source_problem.state.sub(0))
    assigner_p = dolfin.FunctionAssigner(dest_problem.state_space.sub(1), source_problem.state_space.sub(1))
    assigner_p.assign(dest_problem.state.sub(1), source_problem.state.sub(1))


def _compute_ta_target(cavity_problem, coupling=None):
    active = cavity_problem.material.active
    u = get_u_view(cavity_problem)
    F = dolfin.grad(u) + dolfin.Identity(3)
    f = F * active.f0
    lmbda_expr = dolfin.sqrt(f ** 2)

    scratch = dolfin.Function(active.Ta_current.function_space())
    active._projector.project(scratch, active.Ta(lmbda_expr))
    return scratch.vector().get_local().copy()


def _cheap_ta_now(mech_problem):
    """Pure EP-side quantity — no mechanics re-solve. Returns a GLOBAL,
    MPI-reduced (max, min); every rank must agree, since this feeds a
    branch decision gating a collective solve."""
    active = mech_problem.material.active
    scratch = dolfin.Function(active.Ta_current.function_space())
    active._projector.project(scratch, active.Ta(active.lmbda))
    local_vals = scratch.vector().get_local()
    local_max = float(local_vals.max()) if len(local_vals) > 0 else -1e30
    local_min = float(local_vals.min()) if len(local_vals) > 0 else 1e30
    comm = mech_problem.geometry.mesh.mpi_comm()
    return dolfin.MPI.max(comm, local_max), dolfin.MPI.min(comm, local_min)


def solve_cavity_with_ta_ramp(cavity_manager, lv_cavity, rv_cavity, max_ta_step=1.0,
                               max_step_doublings=6, coupling=None):
    problem = cavity_manager.get_problem(lv_cavity, rv_cavity)
    active = problem.material.active

    def _ramp_to(target_vals, label):
        nonlocal problem, active
        Ta_start = active.Ta_current.vector().get_local().copy()
        diff = target_vals - Ta_start
        local_max_jump = float(np.abs(diff).max()) if len(diff) > 0 else 0.0
        max_jump = dolfin.MPI.max(problem.geometry.mesh.mpi_comm(), local_max_jump)
        n_steps = max(1, int(np.ceil(max_jump / max_ta_step)))

        logger.info(f"  [Ta ramp:{label}] Ta_start_max={Ta_start.max():.4f}, "
                    f"Ta_target_max={target_vals.max():.4f}, max_jump={max_jump:.4f}, "
                    f"planned n_steps={n_steps}")

        for attempt in range(max_step_doublings + 1):
            alpha_vals = np.linspace(0.0, 1.0, n_steps + 1)[1:]
            ok = True
            for i, alpha in enumerate(alpha_vals):
                if i > 0:
                    comm = problem.geometry.mesh.mpi_comm()
                    dolfin.MPI.barrier(comm)  # force all ranks to the same point before
                trial_vals = Ta_start + alpha * diff
                active.Ta_current.vector().set_local(trial_vals)
                active.Ta_current.vector().apply("insert")

                try:
                    nliter, nlconv = problem._raw_solve()
                    local_ok = 1 if (nlconv and nliter <= 6) else 0
                except RuntimeError as e:
                    logger.info(f"    [Ta ramp:{label}] substep {i + 1} raised RuntimeError: {e}")
                    local_ok = 0
                    nliter, nlconv = None, False

                comm = problem.geometry.mesh.mpi_comm()
                global_ok = dolfin.MPI.min(comm, local_ok)
                if not global_ok:
                    ok = False
                    logger.info(f"    [Ta ramp:{label}] substep {i + 1} failed/marginal "
                                f"(nlconv={nlconv}, nliter={nliter})")
                    break

                logger.info(f"    [Ta ramp:{label}] substep {i + 1} converged in {nliter} iters")

            if ok:
                break
            logger.info(f"  [Ta ramp:{label}] failed/marginal at n_steps={n_steps} "
                        f"(max_jump={max_jump:.3f}), doubling to {n_steps * 2}")
            n_steps *= 2
        else:
            raise RuntimeError(f"[Ta ramp:{label}] failed even at {n_steps} substeps")

        logger.info(f"  [Ta ramp:{label}] max_jump={max_jump:.3f}, used {n_steps} substep(s)")
        return n_steps

    Ta_target = _compute_ta_target(problem, coupling=coupling)
    n_steps_main = _ramp_to(Ta_target, "main")

    problem._update_active_stress_bookkeeping()
    zetas_max = dolfin.MPI.max(dolfin.MPI.comm_world, float(active._Zetas.vector().max()))
    zetas_min = dolfin.MPI.min(dolfin.MPI.comm_world, float(active._Zetas.vector().min()))
    zetaw_max = dolfin.MPI.max(dolfin.MPI.comm_world, float(active._Zetaw.vector().max()))
    zetaw_min = dolfin.MPI.min(dolfin.MPI.comm_world, float(active._Zetaw.vector().min()))
    logger.info(f"  [Zetas/Zetaw check] Zetas: min={zetas_min:.4f} max={zetas_max:.4f}, "
                f"Zetaw: min={zetaw_min:.4f} max={zetaw_max:.4f}, active.dt={active.dt:.6f}")
    logger.info(f"  [Ta ramp] total substeps this macro step: {n_steps_main}")
    return n_steps_main, problem


class DualCavityModeManager:
    def __init__(self, base_mech_problem, biv_geo, material,
                 lv_marker, rv_marker, lv_pressure_const, rv_pressure_const):
        self.base_problem = base_mech_problem
        self.geometry = biv_geo
        self.material = material
        self.lv_marker = lv_marker
        self.rv_marker = rv_marker
        self.lv_pressure_const = lv_pressure_const
        self.rv_pressure_const = rv_pressure_const
        self.lv_v_target = dolfin.Constant(0.0)
        self.rv_v_target = dolfin.Constant(0.0)
        self._cache = {}
        self._current_key = (False, False)
        self._current_problem = base_mech_problem

    def _build_problem(self, lv_cavity, rv_cavity):
        cavity_specs = {}
        if lv_cavity:
            cavity_specs[self.lv_marker] = self.lv_v_target
        if rv_cavity:
            cavity_specs[self.rv_marker] = self.rv_v_target

        neumann = []
        if not lv_cavity:
            neumann.append(pulse.NeumannBC(traction=self.lv_pressure_const, marker=self.lv_marker))
        if not rv_cavity:
            neumann.append(pulse.NeumannBC(traction=self.rv_pressure_const, marker=self.rv_marker))
        robin = list(self.base_problem.bcs.robin)
        bcs = pulse.BoundaryConditions(neumann=neumann, robin=robin, dirichlet=[])

        return CavityConstrainedMechanicsProblem(
            self.geometry, self.material, bcs, cavity_specs=cavity_specs,
            solver_parameters={"report": True, "absolute_tolerance": 1e-4, "relative_tolerance": 1e-4},
        )

    def get_problem(self, lv_cavity, rv_cavity):
        key = (lv_cavity, rv_cavity)
        if key == (False, False):
            problem = self.base_problem
        elif key not in self._cache:
            problem = self._build_problem(lv_cavity, rv_cavity)
            self._cache[key] = problem
        else:
            problem = self._cache[key]

        if key != self._current_key:
            handoff_uP(self._current_problem, problem)
            self._current_key = key
            self._current_problem = problem
            logger.info(f"  [cavity manager] switched mode: LV_cavity={lv_cavity}, RV_cavity={rv_cavity}")

        return problem