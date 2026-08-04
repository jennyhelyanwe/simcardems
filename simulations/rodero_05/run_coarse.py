import os
from copy import deepcopy

cache_dir = os.environ.get("FENICS_CACHE_DIR", os.path.expanduser("~/.cache"))
os.environ["XDG_CACHE_HOME"] = cache_dir

import logging
logging.getLogger("simcardems.newton_solver").setLevel(logging.DEBUG)
logging.getLogger("simcardems.biv_cavity_cycle_controller").setLevel(logging.INFO)
logging.getLogger("simcardems.runner").setLevel(logging.DEBUG)
logging.getLogger("simcardems.models.fully_coupled_Tor_Land.active_model").setLevel(logging.INFO)
logging.getLogger("simcardems.models.fully_coupled_Tor_Land.em_model").setLevel(logging.DEBUG)
logging.getLogger("__main__").setLevel(logging.DEBUG)

import dataclasses
import json
import time

import dolfin
dolfin.PETScOptions.set("mat_mumps_icntl_4", "0")
dolfin.parameters["reorder_dofs_serial"] = True
dolfin.parameters["mesh_partitioner"] = "ParMETIS"  # instead of default "SCOTCH"
import numpy as np
import pandas as pd
import pulse
from pulse import kinematics as _kinematics
from pulse.dolfin_utils import list_sum as _list_sum
import pulse.mechanicsproblem as _mp
try:
    import ufl_legacy as _ufl
except ImportError:
    import ufl as _ufl

from cardiac_geometries.geometry import Geometry
from simcardems.bivgeometry import BiVentricularGeometry
from simcardems.activation import (
    load_activation_times,
    interpolate_activation_to_ep_mesh,
    endocardial_stimulus_domain,
)
from simcardems.spatial_fields import (
    map_dense_field_to_dg0_function,
    load_dense_node_field,
    map_dense_field_to_ep_mesh,
)
from simcardems.config import Config
from simcardems.models import em_model
from simcardems.models.fully_coupled_Tor_Land.cell_model import TorLandFull
from simcardems.runner import Runner
from simcardems.time_stepper import TimeStepper
from simcardems.biv_cavity_cycle_controller import (
    BiVCycleController,
    CavityState,
    CycleParams,
    WindkesselParams,
    Phase,
    compute_cavity_volume,
    compute_target_pressure,
    advance_phase,
    commit_step,
)
from simcardems.postprocess import ecg_recovery
from simcardems.geometry import refine_mesh, StimulusDomain
from simcardems.steady_state_cell_cache import (
    load_steady_state_cache,
    apply_celltype_aware_initial_conditions,
)
from simcardems import utils

t_script_start = time.time()

logger = utils.getLogger(__name__)
logger.info(['dolfin:', dolfin.__version__])
import petsc4py; logger.info(['petsc4py:', petsc4py.__version__])
from petsc4py import PETSc; logger.info(['PETSc:', PETSc.Sys.getVersion()])

# ── Run-mode toggle ──────────────────────────────────────────────────────
# "local"   : fast, no EP refinement, uniform whole-mesh activation
#             (for iterating on mechanics/cavity-constraint bugs quickly)
# "archer2" : full-fidelity run — 3x EP refinement, real propagating
#             activation from heart.endocardial-activation-times
RUN_MODE = os.environ.get("RUN_MODE", "local")
#RUN_MODE = "local"  # "local" or "archer2"
assert RUN_MODE in ("local", "archer2")

NUM_REFINEMENTS = 3 if RUN_MODE == "archer2" else 0
UNIFORM_ACTIVATION_TEST = (RUN_MODE == "local")

logger.info(f"RUN_MODE = {RUN_MODE}  "
            f"(NUM_REFINEMENTS={NUM_REFINEMENTS}, UNIFORM_ACTIVATION_TEST={UNIFORM_ACTIVATION_TEST})")

RESOLUTION = "4mm"
MESH_DIR = "meshes/"

RUN_TAG = os.environ.get("RUN_TAG", "default")
RESULTS_DIR = f"results_{RESOLUTION}/{RUN_TAG}/"
# ── Warm start configuration ─────────────────────────────────────────────
WARM_START_T_MS = None #  110  # e.g. 100.0 to restart from t=100ms, or None for fresh start


# ── Monkey-patch: normal-only Robin BC (frictionless contact) ───────────
def _external_work_normal_robin(self, u, v):
    F = dolfin.variable(_kinematics.DeformationGradient(u))
    N = self.geometry.facet_normal
    ds = self.geometry.ds
    dx = self.geometry.dx

    external_work = []
    self._debug_traction_parts = {}  # marker -> (kind, N-free UFL piece(s))

    for neumann in self.bcs.neumann:
        n = neumann.traction * _ufl.cofac(F) * N
        self._debug_traction_parts[neumann.marker] = ("neumann", neumann.traction, _ufl.cofac(F))
        external_work.append(dolfin.inner(v, n) * ds(neumann.marker))

    for robin in self.bcs.robin:
        u_normal = dolfin.inner(u, N) * N
        r = robin.value * u_normal
        self._debug_traction_parts[robin.marker] = ("robin", robin.value, u)
        external_work.append(dolfin.inner(r, v) * ds(robin.marker))

    for body_force in self.bcs.body_force:
        external_work.append(-dolfin.derivative(dolfin.inner(body_force, u) * dx, u, v))

    if len(external_work) > 0:
        return _list_sum(external_work)
    return None


_mp.MechanicsProblem._external_work = _external_work_normal_robin
logger.info("Patched MechanicsProblem._external_work: Robin BC normal-only")


# ── Cavity-volume-constrained mechanics (Lagrange multiplier), for IVC/IVR ──
class CavityConstrainedMechanicsProblem(pulse.MechanicsProblem):
    """
    cavity_specs: {marker: v_target_Constant} — one entry per cavity
    CURRENTLY being volume-constrained (gets its own Real multiplier).
    Neumann/Robin BCs passed via `bcs` (e.g. a still-pressure-driven RV,
    or the EPI Robin spring) are handled normally via _external_work.
    """
    def __init__(self, *args, cavity_specs=None, **kwargs):
        if not cavity_specs:
            raise ValueError("cavity_specs must have at least one {marker: v_target}")
        self.cavity_specs = cavity_specs
        super().__init__(*args, **kwargs)

    def _init_spaces(self):
        mesh = self.geometry.mesh
        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        R_elements = [dolfin.FiniteElement("Real", mesh.ufl_cell(), 0)
                      for _ in self.cavity_specs]
        self.state_space = dolfin.FunctionSpace(
            mesh, dolfin.MixedElement([P2, P1] + R_elements)
        )
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
        self._jacobian = dolfin.derivative(
            self._virtual_work, self.state, dolfin.TrialFunction(self.state_space)
        )
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
        if self.strong_coupling:
            active = self.material.active
            u = self.state.split(deepcopy=True)[0]
            F = dolfin.grad(u) + dolfin.Identity(3)
            f = F * active.f0
            lmbda = dolfin.sqrt(f ** 2)
            active._projector.project(active.lmbda, lmbda)
            if active.dt > 0:
                active._projector.project(active._dLambda,
                                           (lmbda - active.lmbda_prev) / active.dt)
            if update_ta_current:
                active._projector.project(active.Ta_current, active.Ta(lmbda))
            active.update_current(lmbda=lmbda)
            active.update_prev()

    def get_cavity_pressure(self, marker):
        idx = list(self.cavity_specs.keys()).index(marker)
        P_fn = self.state.split(deepcopy=True)[2 + idx]
        return get_real_space_value(P_fn)


def handoff_uP(source_problem, dest_problem):
    """Copy (u, p) from source_problem.state into dest_problem.state's
    corresponding sub-blocks — works regardless of extra P_lv/P_rv blocks
    on either side, since we only ever touch sub(0)/sub(1)."""
    parts = source_problem.state.split(deepcopy=True)
    u_src, p_src = parts[0], parts[1]
    dolfin.assign(dest_problem.state.sub(0), u_src)
    dolfin.assign(dest_problem.state.sub(1), p_src)


def _compute_ta_target(cavity_problem):
    """What Ta would be right now, using the last-solved displacement and
    freshly-interpolated XS_mech/XW_mech — the aiming point for the ramp."""
    active = cavity_problem.material.active
    u = cavity_problem.state.split(deepcopy=True)[0]
    F = dolfin.grad(u) + dolfin.Identity(3)
    f = F * active.f0
    lmbda_expr = dolfin.sqrt(f ** 2)
    scratch = dolfin.Function(active.Ta_current.function_space())
    active._projector.project(scratch, active.Ta(lmbda_expr))
    return scratch.vector().get_local().copy()


def solve_cavity_with_ta_ramp(cavity_problem, max_ta_step=2.0, max_step_doublings=6):
    """Ramp Ta_current toward the EP-driven target, then reconcile against
    the lambda the ramp actually produced."""
    active = cavity_problem.material.active

    def _ramp_to(target_vals, label):
        Ta_start = active.Ta_current.vector().get_local().copy()
        diff = target_vals - Ta_start
        max_jump = np.abs(diff).max()
        n_steps = max(1, int(np.ceil(max_jump / max_ta_step)))

        logger.info(f"  [Ta ramp:{label}] Ta_start_max={Ta_start.max():.4f}, "
                    f"Ta_target_max={target_vals.max():.4f}, max_jump={max_jump:.4f}, "
                    f"planned n_steps={n_steps}")

        for attempt in range(max_step_doublings + 1):
            alpha_vals = np.linspace(0.0, 1.0, n_steps + 1)[1:]
            ok = True
            for i, alpha in enumerate(alpha_vals):
                trial_vals = Ta_start + alpha * diff
                active.Ta_current.vector().set_local(trial_vals)
                active.Ta_current.vector().apply("insert")
                try:
                    nliter, nlconv = cavity_problem._raw_solve()
                    if not nlconv or nliter > 6:
                        ok = False
                        logger.info(f"    [Ta ramp:{label}] substep {i+1} failed/marginal "
                                    f"(nlconv={nlconv}, nliter={nliter})")
                        break
                except RuntimeError:
                    ok = False
                    logger.info(f"    [Ta ramp:{label}] substep {i+1} raised RuntimeError")
                    break
            if ok:
                break
            logger.info(f"  [Ta ramp:{label}] failed/marginal at n_steps={n_steps} "
                        f"(max_jump={max_jump:.3f}), doubling to {n_steps * 2}")
            n_steps *= 2
        else:
            raise RuntimeError(f"[Ta ramp:{label}] failed even at {n_steps} substeps")

        logger.info(f"  [Ta ramp:{label}] max_jump={max_jump:.3f}, used {n_steps} substep(s)")
        return n_steps

    Ta_target = _compute_ta_target(cavity_problem)
    n_steps_main = _ramp_to(Ta_target, "main")

    cavity_problem._update_active_stress_bookkeeping(update_ta_current=False)
    Ta_consistent = _compute_ta_target(cavity_problem)
    n_steps_correction = _ramp_to(Ta_consistent, "consistency")

    cavity_problem._update_active_stress_bookkeeping(update_ta_current=False)
    total_steps = n_steps_main + n_steps_correction
    logger.info(f"  [Ta ramp] total substeps this macro step: {total_steps} "
                f"(main={n_steps_main}, consistency={n_steps_correction})")
    return total_steps


class DualCavityModeManager:
    """Tracks, independently, whether LV and RV are each currently
    volume-constrained (isovolumic) or pressure-driven, builds/caches the
    appropriate MechanicsProblem variant for the current combination, and
    hands off (u, p) state whenever the combination changes."""
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


# ── PseudoECG ──────────────────────────────────────────────────────────
class PseudoECG:
    def __init__(self, leads: dict, sigma_b: float = 1.0):
        self.leads = leads
        self.sigma_b = sigma_b
        self.log = {name: [] for name in leads}
        self.times = []
        self._file = None

    def open_file(self, path: str):
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            self._file = open(path, "w")
            self._file.write("t_ms," + ",".join(self.leads.keys()) + "\n")
            self._file.flush()

    def compute(self, v: dolfin.Function, t: float):
        mesh = v.function_space().mesh()
        vals = []
        for name, (pos, neg) in self.leads.items():
            phi_pos = ecg_recovery(v=v, sigma_b=self.sigma_b, point=np.array(pos), mesh=mesh)
            phi_neg = ecg_recovery(v=v, sigma_b=self.sigma_b, point=np.array(neg), mesh=mesh)
            val = phi_pos - phi_neg
            self.log[name].append(val)
            vals.append(val)
        self.times.append(t)
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0 and self._file is not None:
            self._file.write(f"{t:.3f}," + ",".join(f"{v:.8f}" for v in vals) + "\n")
            self._file.flush()

    def close(self):
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0 and self._file is not None:
            self._file.close()


# ── BiVCycleRunner ───────────────────────────────────────────────────────
class BiVCycleRunner(Runner):
    def set_cycle_controller(self, controller, mech_problem, outdir,
                              lv_pressure_const, rv_pressure_const,
                              pseudo_ecg=None, dt_ecg=5.0,
                              warm_start_freq=5.0, t_restart=0.0):
        self._cycle_controller = controller
        self._mech_problem = mech_problem
        self._lv_pressure_const = lv_pressure_const
        self._rv_pressure_const = rv_pressure_const
        self._outdir = outdir
        self._pseudo_ecg = pseudo_ecg
        self._dt_ecg = dt_ecg
        self._last_ecg_t = -dt_ecg
        self._warm_start_freq = warm_start_freq
        self._last_warm_start_t = -warm_start_freq

        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            pv_path = os.path.join(outdir, "pv_loop.csv")
            if os.path.exists(pv_path) and t_restart > 0:
                df = pd.read_csv(pv_path)
                df = df[df['t_ms'] <= t_restart]
                df.to_csv(pv_path, index=False)
                self._pv_file = open(pv_path, "a")
            else:
                self._pv_file = open(pv_path, "w")
                self._pv_file.write("t_ms,LVP_kPa,LVV,RVP_kPa,RVV,LV_phase,RV_phase\n")

            if pseudo_ecg is not None:
                ecg_path = os.path.join(outdir, "pseudo_ecg.csv")
                if os.path.exists(ecg_path) and t_restart > 0:
                    df = pd.read_csv(ecg_path)
                    df = df[df['t_ms'] <= t_restart]
                    df.to_csv(ecg_path, index=False)
                    pseudo_ecg._file = open(ecg_path, "a")
                else:
                    pseudo_ecg.open_file(ecg_path)
        else:
            self._pv_file = None

    def _solve_mechanics(self):
        self.coupling.coupling_to_mechanics()
        t0 = time.time()

        lv_state = self._cycle_controller.lv_state
        rv_state = self._cycle_controller.rv_state
        isovol_phases = (Phase.ISOVOL_CONTRACTION, Phase.ISOVOL_RELAXATION)
        lv_isovol = lv_state.phase in isovol_phases
        rv_isovol = rv_state.phase in isovol_phases

        t_ms = TimeStepper.ns2ms(self.t)
        dt_ms = self._config.dt

        logger.info(f"  [dual-cavity] t={t_ms:.2f}ms  LV_phase={lv_state.phase} "
                    f"(isovol={lv_isovol})  RV_phase={rv_state.phase} (isovol={rv_isovol})")

        active_problem = self._cavity_manager.get_problem(lv_isovol, rv_isovol)
        logger.info(f"  [dual-cavity] using problem key={self._cavity_manager._current_key} "
                    f"(is_base_problem={active_problem is self._cavity_manager.base_problem})")

        if lv_isovol:
            target = lv_state.end_dia_vol if lv_state.phase == Phase.ISOVOL_CONTRACTION else lv_state.end_sys_vol
            self._cavity_manager.lv_v_target.assign(target)
            logger.info(f"  [dual-cavity] LV target_vol={target:.1f} mm^3 (Lagrange multiplier mode)")
        if rv_isovol:
            target = rv_state.end_dia_vol if rv_state.phase == Phase.ISOVOL_CONTRACTION else rv_state.end_sys_vol
            self._cavity_manager.rv_v_target.assign(target)
            logger.info(f"  [dual-cavity] RV target_vol={target:.1f} mm^3 (Lagrange multiplier mode)")

        if not lv_isovol:
            u_now = active_problem.state.split(deepcopy=True)[0]
            v_lv_now = compute_cavity_volume(self._cycle_controller.geometry, u_now,
                                              self._cycle_controller.lv_marker)
            p_lv_target = compute_target_pressure(lv_state, v_lv_now, t_ms, dt_ms)
            self._lv_pressure_const.assign(p_lv_target)
            logger.info(f"  [dual-cavity] LV pressure-driven: v_lv_now={v_lv_now:.1f}, "
                        f"p_lv_target={p_lv_target:.4f} kPa (Neumann BC mode)")
        if not rv_isovol:
            u_now = active_problem.state.split(deepcopy=True)[0]
            v_rv_now = compute_cavity_volume(self._cycle_controller.geometry, u_now,
                                              self._cycle_controller.rv_marker)
            p_rv_target = compute_target_pressure(rv_state, v_rv_now, t_ms, dt_ms)
            self._rv_pressure_const.assign(p_rv_target)
            logger.info(f"  [dual-cavity] RV pressure-driven: v_rv_now={v_rv_now:.1f}, "
                        f"p_rv_target={p_rv_target:.4f} kPa (Neumann BC mode)")

        if lv_isovol or rv_isovol:
            n_substeps = solve_cavity_with_ta_ramp(active_problem)
            logger.info(f"  [dual-cavity] cavity-constrained solve done, {n_substeps} total Ta-ramp substeps")
        else:
            active_problem.solve()
            logger.info(f"  [dual-cavity] plain pressure-driven solve done")
        u_final = active_problem.state.split(deepcopy=True)[0]

        self.coupling.interpolate(self.coupling.lmbda_mech, self.coupling.lmbda_ep)

        v_lv_final = compute_cavity_volume(self._cycle_controller.geometry, u_final,
                                            self._cycle_controller.lv_marker)
        v_rv_final = compute_cavity_volume(self._cycle_controller.geometry, u_final,
                                            self._cycle_controller.rv_marker)
        p_lv_final = (active_problem.get_cavity_pressure(self._cycle_controller.lv_marker)
                      if lv_isovol else float(self._lv_pressure_const))
        p_rv_final = (active_problem.get_cavity_pressure(self._cycle_controller.rv_marker)
                      if rv_isovol else float(self._rv_pressure_const))

        logger.info(f"  [dual-cavity] FINAL this step: LV p={p_lv_final:.4f} kPa v={v_lv_final:.1f} mm^3  |  "
                    f"RV p={p_rv_final:.4f} kPa v={v_rv_final:.1f} mm^3")

        commit_step(lv_state, v_lv_final, p_lv_final)
        commit_step(rv_state, v_rv_final, p_rv_final)
        # NOTE (flagged, not removed): these two lines unconditionally
        # overwrite end_dia_vol every step, not just during PRELOAD.
        # Confirm this is intentional — compute_target_pressure's own
        # PRELOAD branch already sets end_dia_vol correctly using
        # volume_iter_k; this looks like a leftover from earlier
        # debugging rather than something we deliberately kept.
        lv_state.end_dia_vol = v_lv_final
        rv_state.end_dia_vol = v_rv_final

        lv_phase_before = lv_state.phase
        rv_phase_before = rv_state.phase
        advance_phase(lv_state, t_ms, dt_ms)
        advance_phase(rv_state, t_ms, dt_ms)

        if lv_state.phase != lv_phase_before:
            logger.info(f"  [dual-cavity] *** LV PHASE TRANSITION: {lv_phase_before} -> {lv_state.phase} *** "
                        f"(pressure_n={lv_state.pressure_n:.4f}, wdk_pressure_n={lv_state.wdk_pressure_n:.4f})")
        if rv_state.phase != rv_phase_before:
            logger.info(f"  [dual-cavity] *** RV PHASE TRANSITION: {rv_phase_before} -> {rv_state.phase} *** "
                        f"(pressure_n={rv_state.pressure_n:.4f}, wdk_pressure_n={rv_state.wdk_pressure_n:.4f})")

        self.coupling.update_prev_mechanics()
        self.coupling.mechanics_to_coupling()
        self.coupling.coupling_to_ep()

        apply_phase_dt(self._config, lv_state.phase, time_stepper=self._time_stepper, logger=logger)

        t1 = time.time()
        logger.debug(f"  Mechanics solve time: {t1 - t0:.2f}s")
        logger.debug(
            f"  → t={t_ms:.1f} ms"
            f"  LV phase={lv_state.phase}  LVP={lv_state.pressure_n:.3f} kPa  LVV={lv_state.volume_n / 1000:.2f} mL"
            f"  RV phase={rv_state.phase}  RVP={rv_state.pressure_n:.3f} kPa  RVV={rv_state.volume_n / 1000:.2f} mL"
        )

        t_ms_now = TimeStepper.ns2ms(self.t)
        if (t_ms_now - self._last_warm_start_t) >= (self._warm_start_freq - 1e-10):
            checkpoint_name = f"warm_start_{int(t_ms_now):04d}ms"
            with dolfin.HDF5File(dolfin.MPI.comm_world, os.path.join(self._outdir, f"{checkpoint_name}.h5"), "w") as f:
                f.write(self.coupling.ep_solver.vs, "/ep/vs")
                f.write(self.coupling.mech_solver.state, "/mechanics/state")
                f.write(self.coupling.lmbda_mech, "/em/lmbda_prev")
                f.write(self.coupling.Zetas_mech, "/em/Zetas_prev")
                f.write(self.coupling.Zetaw_mech, "/em/Zetaw_prev")
            if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
                with open(os.path.join(self._outdir, f"{checkpoint_name}.json"), "w") as f:
                    json.dump({
                        "t_ms": t_ms_now,
                        "lv": {k: v for k, v in dataclasses.asdict(lv_state).items() if not isinstance(v, dict)},
                        "rv": {k: v for k, v in dataclasses.asdict(rv_state).items() if not isinstance(v, dict)},
                    }, f, indent=2)
            self._last_warm_start_t = t_ms_now

        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0 and self._pv_file is not None:
            self._pv_file.write(
                f"{t_ms_now:.3f},{lv_state.pressure_n:.6f},{lv_state.volume_n:.6f},"
                f"{rv_state.pressure_n:.6f},{rv_state.volume_n:.6f},"
                f"{lv_state.phase},{rv_state.phase}\n"
            )
            self._pv_file.flush()

        if self._pseudo_ecg is not None and t_ms_now - self._last_ecg_t >= self._dt_ecg - 1e-10:
            self.coupling.assigners.assign_ep()
            v_fn = self.coupling.assigners.functions["ep"]["V"]
            self._pseudo_ecg.compute(v_fn, t_ms_now)
            self._last_ecg_t = t_ms_now

    def close_files(self):
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0 and self._pv_file is not None:
            self._pv_file.close()
        if self._pseudo_ecg is not None:
            self._pseudo_ecg.close()


# ── Sanity checks / helpers ─────────────────────────────────────────────
def check_orthonormality(f0, s0, n0, label=""):
    def as_array(x):
        if hasattr(x, "vector"):
            return x.vector().get_local().reshape(-1, 3)
        return x

    F = as_array(f0)
    S = as_array(s0)
    N = as_array(n0)

    f0n = np.linalg.norm(F, axis=1)
    dot_fs = np.abs(np.sum(F * S, axis=1))
    dot_fn = np.abs(np.sum(F * N, axis=1))
    dot_sn = np.abs(np.sum(S * N, axis=1))
    logger.info(f"{label} f0 norm: {f0n.min():.6f}-{f0n.max():.6f}, "
                f"max|f0.s0|={dot_fs.max():.3e}, max|f0.n0|={dot_fn.max():.3e}, max|s0.n0|={dot_sn.max():.3e}")


def get_real_space_value(fn):
    """Safely extract the single global scalar value of a Real-space
    Function under MPI — exactly one rank owns the DOF locally; every
    other rank has an empty local array. Summing local contributions
    across ranks (only one is ever nonzero) via MPI.sum gives a
    consistent, correct value on every rank."""
    local = fn.vector().get_local()
    local_val = float(local[0]) if len(local) > 0 else 0.0
    comm = fn.function_space().mesh().mpi_comm()
    return dolfin.MPI.sum(comm, local_val)


def tet_volumes(coords, cells_arr):
    p0 = coords[cells_arr[:, 0]]
    p1 = coords[cells_arr[:, 1]]
    p2 = coords[cells_arr[:, 2]]
    p3 = coords[cells_arr[:, 3]]
    return np.abs(np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0


def tet_quality_ratios(coords, cells_arr):
    p0 = coords[cells_arr[:, 0]]
    p1 = coords[cells_arr[:, 1]]
    p2 = coords[cells_arr[:, 2]]
    p3 = coords[cells_arr[:, 3]]

    vol = np.abs(np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0

    def tri_area(a, b, c):
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    A0 = tri_area(p1, p2, p3)
    A1 = tri_area(p0, p2, p3)
    A2 = tri_area(p0, p1, p3)
    A3 = tri_area(p0, p1, p2)
    surface_area = A0 + A1 + A2 + A3
    inradius = 3 * vol / surface_area

    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p0, axis=1)
    c = np.linalg.norm(p3 - p0, axis=1)
    a1 = np.linalg.norm(p3 - p2, axis=1)
    b1 = np.linalg.norm(p3 - p1, axis=1)
    c1 = np.linalg.norm(p2 - p1, axis=1)

    num = np.sqrt(
        (a * a1 + b * b1 + c * c1) *
        (a * a1 + b * b1 - c * c1) *
        (a * a1 - b * b1 + c * c1) *
        (-a * a1 + b * b1 + c * c1)
    )
    circumradius = num / (24 * vol)
    return inradius / circumradius


# ── Phase-dependent time steps dt/dt_mech ────────────────────────────────
def dt_targets_for_phase(phase):
    """Returns the intended (dt, dt_mech) in ms for a given cardiac phase.
    Edit values HERE ONLY — both the pre-loop setup and the in-loop
    phase-change check call this same function, so they can't drift
    out of sync with each other."""
    if phase == Phase.PRELOAD:
        return dict(dt=10.0, dt_mech=10.0)
    else:
        return dict(dt=0.1, dt_mech=5)


def apply_phase_dt(config, phase, time_stepper=None, logger=None):
    """Applies dt_targets_for_phase(phase) to config, and — if a live
    TimeStepper is passed — also syncs it, with the required ms->ns
    conversion (TimeStepper stores dt in ns after construction; its
    setter does NOT convert for you)."""
    targets = dt_targets_for_phase(phase)
    changed = (config.dt != targets["dt"]) or (config.dt_mech != targets["dt_mech"])
    if changed and logger is not None:
        logger.info(f"dt update (phase={phase}): "
                    f"dt {config.dt} -> {targets['dt']}, "
                    f"dt_mech {config.dt_mech} -> {targets['dt_mech']}")
    config.dt = targets["dt"]
    config.dt_mech = targets["dt_mech"]
    if time_stepper is not None:
        time_stepper.dt = TimeStepper.ms2ns(targets["dt"])
    return changed


# ── 1. Geometry ───────────────────────────────────────────────────────────
logger.info('Build geometry...')

geo = Geometry.from_file(MESH_DIR + "rodero_05_coarse_" + RESOLUTION + ".h5")

mesh = dolfin.Mesh()
with dolfin.HDF5File(mesh.mpi_comm(), MESH_DIR + "rodero_05_coarse_" + RESOLUTION + ".h5", "r") as f:
    f.read(mesh, "mesh", False)

coords = mesh.coordinates()
cells_arr = mesh.cells()
logger.info(f"Mesh: {mesh.num_vertices()} vertices, {mesh.num_cells()} cells")

volumes = tet_volumes(coords, cells_arr)
logger.debug(f"Cell volumes: min={volumes.min():.4f}, mean={volumes.mean():.4f}, max={volumes.max():.4f}")
logger.debug(f"Negative volumes: {np.sum(volumes < 0)}")
logger.debug(f"Very small volumes (<0.01): {np.sum(volumes < 0.01)}")

radii = tet_quality_ratios(coords, cells_arr)
logger.debug(f"Inradius/circumradius: min={radii.min():.4f}, mean={radii.mean():.4f}")
logger.debug(f"Poor quality (ratio < 0.1): {np.sum(radii < 0.1)}")
logger.debug(f"Poor quality (ratio < 0.05): {np.sum(radii < 0.05)}")
logger.debug(f"Poor quality (ratio < 0.02): {np.sum(radii < 0.02)}")

# Build refined EP mesh with parent tracking (RUN_MODE-dependent)
if NUM_REFINEMENTS > 1:
    ep_mesh = refine_mesh(geo.mesh, num_refinements=NUM_REFINEMENTS)
    ffun_ep = dolfin.adapt(geo.ffun, ep_mesh)
    biv_geo = BiVentricularGeometry.from_geometry(
        geo, ep_mesh=ep_mesh, ffun_ep=ffun_ep,
        parameters={"num_refinements": NUM_REFINEMENTS},
    )
else:
    biv_geo = BiVentricularGeometry.from_geometry(geo, ep_mesh=geo.mesh, ffun_ep=geo.ffun)

logger.info(f"EP Mesh vertices: {biv_geo.ep_mesh.num_vertices()}")
logger.info(f"Mechanics Mesh vertices: {biv_geo.mechanics_mesh.num_vertices()}")

# Load valve plug mask
coarse_tv = np.load(MESH_DIR + '/rodero_05_coarse_' + RESOLUTION + '_tv.npy')
is_valve_float = (coarse_tv >= 7).astype(float)
coarse_centres_all = np.array([cell.midpoint().array()
                                for cell in dolfin.cells(biv_geo.mechanics_mesh)])
valve_fn = map_dense_field_to_ep_mesh(biv_geo.mechanics_mesh, coarse_centres_all, is_valve_float)
logger.debug(f'Mechanics valve plug elements: {int(is_valve_float.sum())}')
biv_geo.valve_mask = valve_fn

check_orthonormality(biv_geo.f0, biv_geo.s0, biv_geo.n0)

epi_marker = geo.markers["EPI"][0]
base_marker = geo.markers["BASE"][0]
ffun_arr = geo.ffun.array()
logger.debug(f"EPI marker {epi_marker}: {np.sum(ffun_arr == epi_marker)} facets")
logger.debug(f"BASE marker {base_marker}: {np.sum(ffun_arr == base_marker)} facets")
logger.debug(f"Total facets: {len(ffun_arr)}")

f0_arr = biv_geo.f0.vector().get_local().reshape(-1, 3)
s0_arr = biv_geo.s0.vector().get_local().reshape(-1, 3)
dot_fs = np.sum(f0_arr * s0_arr, axis=1)
logger.debug(f"f0.s0 range on 4mm mesh: {dot_fs.min():.6f} to {dot_fs.max():.6f}")
logger.debug(f"Number of cells with |f0.s0| > 0.5: {np.sum(np.abs(dot_fs) > 0.5)}")


# ── 2. Activation times ───────────────────────────────────────────────────
logger.info('Load activation times...')
node_coords = pd.read_csv(MESH_DIR + "/rodero_05_fine_xyz.csv", header=None).to_numpy() * 10.0

node_ids, activation_times, coords_subset = load_activation_times(
    MESH_DIR + "/heart.endocardial-activation-times", node_coords
)
activation_times = activation_times * 1000.0  # s -> ms

act_fn = interpolate_activation_to_ep_mesh(
    ep_mesh=biv_geo.ep_mesh,
    endo_marker_ep=[biv_geo.markers["ENDO_LV"][0], biv_geo.markers["ENDO_RV"][0]],
    ffun_ep=biv_geo.ffun_ep,
    node_ids_mech=node_ids,
    activation_times_mech=activation_times,
    coords_subset_mech=coords_subset,
)

# ── 3. Stimulus domain (RUN_MODE-dependent) ───────────────────────────────
if UNIFORM_ACTIVATION_TEST:
    logger.info('Uniform whole-mesh activation (local/debug mode)...')
    act_fn.vector()[:] = 110
    logger.info("DEBUG: uniform activation override — every node activates at t=110 ms")

    whole_mesh_marker = dolfin.MeshFunction("size_t", biv_geo.ep_mesh,
                                              biv_geo.ep_mesh.topology().dim(), 1)
    biv_geo.stimulus_domain = StimulusDomain(domain=whole_mesh_marker, marker=1)
else:
    logger.info('Real endocardial stimulus domain (archer2/full-fidelity mode)...')
    biv_geo.stimulus_domain = endocardial_stimulus_domain(
        mesh=biv_geo.ep_mesh,
        ffun=biv_geo.ffun_ep,
        endo_markers=[biv_geo.markers["ENDO_LV"][0], biv_geo.markers["ENDO_RV"][0]],
        layer_thickness=2,
    )

# ── Load cache or pace cells to steady state ──────────────────────────────
CELL_PARAMS_OVERRIDE = {}
PCL = 800.0

steady_state = load_steady_state_cache(CELL_PARAMS_OVERRIDE, PCL, max_beats=300)
cell_init_file = steady_state["cell_init_files"][0]  # endo, matching original default

# ── 4. Config ──────────────────────────────────────────────────────────────
logger.info('Configuring...')
config = Config()
config.cell_init_file = cell_init_file
config.T = 800.0
config.dt = 10.0
config.dt_mech = 10.0
config.geometry_path = MESH_DIR + "rodero_05_coarse_" + RESOLUTION + ".h5"
config.outdir = RESULTS_DIR + "biv_coarse_run_output"
config.coupling_type = "fully_coupled_Tor_Land"
config.save_freq = 10  # ms
assert config.save_freq >= config.dt
assert config.save_freq >= config.dt_mech
config.linear_mechanics_solver = "mumps"
config.spring = 50.0
config.traction = 0.005
config.mechanics_use_custom_newton_solver = True
config.mechanics_solve_strategy = "fixed"
config.mech_threshold = 1.0
config.relaxation_factor = 1.0

SCALABILITY_TEST = os.environ.get("SCALABILITY_TEST", "0") == "1"
if SCALABILITY_TEST:
    config.T = 20.0
    config.save_freq = 1000.0

# Reverted (transversely isotropic) material parameters — known-good baseline.
material_params_override = dict(
    a=2.28, a_f=1.686, b=9.726, b_f=15.779,
    a_s=0.0, b_s=0.0, a_fs=0.0, b_fs=0.0,
)

# ── 5. Spatial fields ────────────────────────────────────────────────────
logger.info('Load cell type...')
ct_values = load_dense_node_field(MESH_DIR + "rodero_05_fine_nodefield_cell-type.csv")
cell_fn = map_dense_field_to_dg0_function(
    biv_geo.ep_mesh, node_coords, ct_values, {1: 0, 2: 2, 3: 1}
)

logger.info('Load sf IKs...')
iks_values = load_dense_node_field(MESH_DIR + "rodero_05_fine_nodefield_sf_IKs.csv")
iks_fn = map_dense_field_to_ep_mesh(biv_geo.ep_mesh, node_coords, iks_values)

# ── 6. EM coupling ───────────────────────────────────────────────────────
VALVE_STIFFNESS_SCALE = 3.0
logger.info("=" * 60)
logger.info("RUN PARAMETERS")
logger.info("=" * 60)
logger.info(f"Run mode:                  {RUN_MODE}")
logger.info(f"Resolution:                {RESOLUTION}")
logger.info(f"EP refinements:            {NUM_REFINEMENTS}")
logger.info(f"Uniform activation test:   {UNIFORM_ACTIVATION_TEST}")
logger.info(f"T (total time):            {config.T} ms")
logger.info(f"Initial dt (EP):           {config.dt} ms")
logger.info(f"Initial dt_mech:           {config.dt_mech} ms")
logger.info(f"Coupling type:             {config.coupling_type}")
logger.info(f"Linear mechanics solver:   {config.linear_mechanics_solver}")
logger.info(f"Custom Newton solver:      {config.mechanics_use_custom_newton_solver}")
logger.info(f"Mechanics solve strategy:  {config.mechanics_solve_strategy}")
logger.info(f"Mech threshold:            {config.mech_threshold}")
logger.info(f"Relaxation factor:         {config.relaxation_factor}")
logger.info(f"Spring (EPI Robin):        {config.spring}")
logger.info(f"Traction (initial):        {config.traction}")
logger.info(f"Valve stiffness scale:     {VALVE_STIFFNESS_SCALE}")
logger.info("-" * 60)
logger.info("Material parameters:")
for k, v in material_params_override.items():
    logger.info(f"  {k:6s} = {v}")
logger.info("-" * 60)
logger.info(f"Save frequency:            {config.save_freq}")
logger.info(f"Output directory:          {config.outdir}")
logger.info(f"Warm start:                {WARM_START_T_MS}")
logger.info("=" * 60)

logger.info('Setting up EM model...')

coupling = em_model.setup_EM_model_from_config(
    config, geometry=biv_geo, activation_times=act_fn,
    celltype_function=cell_fn, iks_scale_function=iks_fn,
    material_parameters=material_params_override,
    valve_stiffness_scale=VALVE_STIFFNESS_SCALE,
)

mech_problem = coupling.mech_solver

state_names = list(TorLandFull.default_initial_conditions().keys())
apply_celltype_aware_initial_conditions(coupling, cell_fn, steady_state["cell_init_files"], state_names)

# ── Option to fix the apex for easier mechanics convergence ──────────────
APPLY_APEX_PIN = False
if APPLY_APEX_PIN:
    epi_marker_val = biv_geo.markers["EPI"][0]
    base_marker_val = biv_geo.markers["BASE"][0]

    mesh_ = biv_geo.mechanics_mesh
    mesh_.init(2, 0)
    ffun_arr_ = biv_geo.ffun.array()

    base_vertex_ids = set()
    epi_vertex_ids = set()
    conn20 = mesh_.topology()(2, 0)
    for fidx in range(mesh_.num_entities(2)):
        if ffun_arr_[fidx] == base_marker_val:
            base_vertex_ids.update(conn20(fidx))
        elif ffun_arr_[fidx] == epi_marker_val:
            epi_vertex_ids.update(conn20(fidx))

    coords_ = mesh_.coordinates()
    base_centroid = coords_[list(base_vertex_ids)].mean(axis=0)
    epi_coords = coords_[list(epi_vertex_ids)]
    dists = np.linalg.norm(epi_coords - base_centroid, axis=1)
    apex_vertex_local = list(epi_vertex_ids)[int(np.argmax(dists))]
    apex_point = coords_[apex_vertex_local]
    logger.info(f"Apex point identified at {apex_point} (farthest EPI vertex from base centroid)")

    def apex_dirichlet_bc(W):
        class ApexPoint(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return dolfin.near(x[0], apex_point[0], 1e-6) and \
                       dolfin.near(x[1], apex_point[1], 1e-6) and \
                       dolfin.near(x[2], apex_point[2], 1e-6)
        V = W.sub(0)
        return [dolfin.DirichletBC(V, dolfin.Constant((0.0, 0.0, 0.0)),
                                    ApexPoint(), method="pointwise")]

    mech_problem.bcs.dirichlet = list(mech_problem.bcs.dirichlet) + [apex_dirichlet_bc]
    mech_problem._set_dirichlet_bc()
    mech_problem._init_solver()
    logger.info("Apex point pinned (Dirichlet, all 3 components); solver rebuilt.")

# ── 7. Cycle controller ───────────────────────────────────────────────────
LV_ENDO_MARKER = biv_geo.markers["ENDO_LV"][0]
RV_ENDO_MARKER = biv_geo.markers["ENDO_RV"][0]

lv_pressure_const = None
rv_pressure_const = None
for nbc in mech_problem.bcs.neumann:
    logger.debug(f"Found Neumann BC: marker={nbc.marker} traction={float(nbc.traction):.6e}")
    if nbc.marker == LV_ENDO_MARKER:
        lv_pressure_const = nbc.traction
    elif nbc.marker == RV_ENDO_MARKER:
        rv_pressure_const = nbc.traction

logger.debug(f"Total Neumann BCs: {len(mech_problem.bcs.neumann)}")

if lv_pressure_const is None:
    raise RuntimeError("No Neumann BC found on ENDO_LV")
if rv_pressure_const is None:
    raise RuntimeError("No Neumann BC found on ENDO_RV")

lv_params = CycleParams(
    t_zero=50.0, t_prestress=0.0, preload_pressure=0.5, prestress_pressure=0.0,
    t_end_diastole=120.0, p_end_diastole=1.0,
    gain_contraction=(0.01, 10), gain_relaxation=(0.05, 0.01),
    p_fill=0.1, period=800.0,
    windkessel=WindkesselParams(p_init=9.0, compliance=1.0, resistance=100.0, evolve=True),
)

rv_params = CycleParams(
    t_zero=50.0, t_prestress=0.0, preload_pressure=0.17, prestress_pressure=0.0,
    t_end_diastole=120.0, p_end_diastole=0.33,
    gain_contraction=(0.01, 10), gain_relaxation=(0.5, 0.2),
    p_fill=0.033, period=800.0,
    windkessel=WindkesselParams(p_init=3.0, compliance=3.0, resistance=33.0, evolve=True),
)

lv_state = CavityState(name="LV", params=lv_params)
rv_state = CavityState(name="RV", params=rv_params)

cycle_controller = BiVCycleController(
    lv_state=lv_state, rv_state=rv_state,
    lv_pressure_constant=lv_pressure_const, rv_pressure_constant=rv_pressure_const,
    geometry=biv_geo, lv_marker=LV_ENDO_MARKER, rv_marker=RV_ENDO_MARKER,
)

u0, _ = mech_problem.state.split(deepcopy=True)
cycle_controller.initialize(u0)
cavity_manager = DualCavityModeManager(
    mech_problem, biv_geo, mech_problem.material,
    LV_ENDO_MARKER, RV_ENDO_MARKER, lv_pressure_const, rv_pressure_const,
)
logger.debug(f"Initial LV volume: {lv_state.volume_n:.2f}")
logger.debug(f"Initial RV volume: {rv_state.volume_n:.2f}")

# ── 8. Pseudo-ECG ─────────────────────────────────────────────────────────
logger.info('Loading electrode locations...')
electrode_df = pd.read_csv(
    MESH_DIR + "rodero_05_fine_nodefield_electrode_xyz.csv",
    header=None, names=["x", "y", "z"],
)
electrodes = electrode_df.to_numpy()

LA, RA, LL = electrodes[0], electrodes[1], electrodes[2]
V1, V2, V3, V4, V5, V6 = electrodes[4], electrodes[5], electrodes[6], \
                          electrodes[7], electrodes[8], electrodes[9]
wct = tuple((LA + RA + LL) / 3.0)

pseudo_ecg = PseudoECG(
    leads={
        "I": (tuple(LA), tuple(RA)),
        "II": (tuple(LL), tuple(RA)),
        "III": (tuple(LL), tuple(LA)),
        "aVR": (tuple(RA), tuple((LA + LL) / 2.0)),
        "aVL": (tuple(LA), tuple((RA + LL) / 2.0)),
        "aVF": (tuple(LL), tuple((RA + LA) / 2.0)),
        "V1": (tuple(V1), wct), "V2": (tuple(V2), wct), "V3": (tuple(V3), wct),
        "V4": (tuple(V4), wct), "V5": (tuple(V5), wct), "V6": (tuple(V6), wct),
    },
    sigma_b=1.0,
)

# ── 9. Run ────────────────────────────────────────────────────────────────
os.makedirs(config.outdir, exist_ok=True)
config.traction = 0.0

runner = BiVCycleRunner.from_models(coupling=coupling, config=config)

t_restart = 0.0
if WARM_START_T_MS is not None:
    warm_start_name = f"warm_start_{int(WARM_START_T_MS):04d}ms"
    WARM_START = os.path.join(config.outdir, warm_start_name)
    with open(WARM_START + ".json") as f:
        ws = json.load(f)
    t_restart = ws["t_ms"]

runner.set_cycle_controller(
    cycle_controller, mech_problem, config.outdir,
    lv_pressure_const, rv_pressure_const,
    pseudo_ecg=pseudo_ecg, dt_ecg=5.0,
    warm_start_freq=5.0, t_restart=t_restart,
)
runner._cavity_manager = cavity_manager

if WARM_START_T_MS is not None:
    warm_start_name = f"warm_start_{int(WARM_START_T_MS):04d}ms"
    WARM_START = os.path.join(config.outdir, warm_start_name)
    if os.path.exists(WARM_START + ".h5") and os.path.exists(WARM_START + ".json"):
        logger.info(f"Loading warm start from t={WARM_START_T_MS:.1f} ms...")
        with dolfin.HDF5File(dolfin.MPI.comm_world, WARM_START + ".h5", "r") as f:
            f.read(coupling.ep_solver.vs, "/ep/vs")
            f.read(coupling.mech_solver.state, "/mechanics/state")
            f.read(coupling.lmbda_mech, "/em/lmbda_prev")
            f.read(coupling.Zetas_mech, "/em/Zetas_prev")
            f.read(coupling.Zetaw_mech, "/em/Zetaw_prev")
        with open(WARM_START + ".json") as f:
            ws = json.load(f)
        for k, v in ws["lv"].items():
            if hasattr(lv_state, k): setattr(lv_state, k, v)
        for k, v in ws["rv"].items():
            if hasattr(rv_state, k): setattr(rv_state, k, v)
        lv_pressure_const.assign(lv_state.pressure_n)
        rv_pressure_const.assign(rv_state.pressure_n)
        runner._t0 = t_restart
        logger.info(f"Restarting from t={t_restart:.1f} ms")
    else:
        logger.info(f"WARNING: warm start file not found for t={WARM_START_T_MS:.1f} ms")

apply_phase_dt(config, cycle_controller.lv_state.phase, time_stepper=None, logger=logger)

try:
    runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=False)
except Exception as e:
    logger.info(f"Runner error: {e}")
finally:
    runner.close_files()
    logger.info("Done.")

t_script_end = time.time()
logger.info(f"TOTAL SCRIPT WALLCLOCK TIME: {t_script_end - t_script_start:.2f}s "
            f"({(t_script_end - t_script_start)/60:.2f} min) "
            f"at {dolfin.MPI.size(dolfin.MPI.comm_world)} ranks")
