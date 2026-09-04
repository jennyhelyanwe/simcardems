"""
run_lv_ellipsoid_isovol_lagrange.py
------------------------------------
Minimal LV-only reproducer for the VecAssemblyEnd/error-77 crash seen
in the full BiV run — cavity-constrained (Lagrange-multiplier) mechanics
forced on from the start, using the SAME split-free accessors and
collective-safe ramp logic as run_coarse.py, so this is a fair,
apples-to-apples isolation test.
"""

import os
cache_dir = os.environ.get("FENICS_CACHE_DIR", os.path.expanduser("~/.cache"))
os.environ["XDG_CACHE_HOME"] = cache_dir

import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import dolfin
dolfin.set_log_level(dolfin.LogLevel.DEBUG)
dolfin.PETScOptions.set("snes_monitor")

import pulse
from pulse import kinematics
from pulse.dolfin_utils import list_sum as _list_sum
try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

import cardiac_geometries
import numpy as np

from simcardems.geometry import StimulusDomain
from simcardems.lvgeometry import LeftVentricularGeometry
from simcardems.activation import interpolate_activation_to_ep_mesh
from simcardems.config import Config
from simcardems.models import em_model
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
    commit_step,
    advance_phase,
)

logger = logging.getLogger(__name__)


def mpi_print(*args, **kwargs):
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print(*args, **kwargs, flush=True)


# ── Split-free accessors — same mechanism as run_coarse.py ─────────────────

def get_u_view(problem):
    if getattr(problem, '_u_standalone', None) is None:
        V_full = problem.state_space
        mesh = V_full.mesh()
        P2 = V_full.ufl_element().sub_elements()[0]
        V_u = dolfin.FunctionSpace(mesh, P2)
        problem._u_standalone = dolfin.Function(V_u)
        problem._u_assigner = dolfin.FunctionAssigner(V_u, V_full.sub(0))
    problem._u_assigner.assign(problem._u_standalone, problem.state.sub(0))
    return problem._u_standalone


def get_cavity_pressure(problem):
    if not hasattr(problem, '_p_standalone'):
        V_full = problem.state_space
        mesh = V_full.mesh()
        R = V_full.ufl_element().sub_elements()[2]
        V_r = dolfin.FunctionSpace(mesh, R)
        problem._p_standalone = dolfin.Function(V_r)
        problem._p_assigner = dolfin.FunctionAssigner(V_r, V_full.sub(2))
    problem._p_assigner.assign(problem._p_standalone, problem.state.sub(2))
    local = problem._p_standalone.vector().get_local()
    local_val = float(local[0]) if len(local) > 0 else 0.0
    comm = problem.geometry.mesh.mpi_comm()
    return dolfin.MPI.sum(comm, local_val)


def handoff_uP(source_problem, dest_problem):
    assigner_u = dolfin.FunctionAssigner(dest_problem.state_space.sub(0), source_problem.state_space.sub(0))
    assigner_u.assign(dest_problem.state.sub(0), source_problem.state.sub(0))
    assigner_p = dolfin.FunctionAssigner(dest_problem.state_space.sub(1), source_problem.state_space.sub(1))
    assigner_p.assign(dest_problem.state.sub(1), source_problem.state.sub(1))


# ── CavityConstrainedMechanicsProblem ───────────────────────────────────────

class CavityConstrainedMechanicsProblem(pulse.MechanicsProblem):
    def __init__(self, *args, lv_marker=None, v_target=None, **kwargs):
        if lv_marker is None:
            raise ValueError("Must supply lv_marker (the ENDO facet marker int)")
        self.lv_marker = lv_marker
        self.v_target = v_target if v_target is not None else dolfin.Constant(0.0)
        self._u_standalone = None
        self._u_assigner = None
        super().__init__(*args, **kwargs)

    def _init_spaces(self):
        logger.debug("Initialize spaces for cavity-constrained mechanics problem")
        mesh = self.geometry.mesh
        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        R = dolfin.FiniteElement("Real", mesh.ufl_cell(), 0)
        self.state_space = dolfin.FunctionSpace(mesh, dolfin.MixedElement([P2, P1, R]))
        self.state = dolfin.Function(self.state_space, name="state")
        self.state_test = dolfin.TestFunction(self.state_space)

    def _init_forms(self):
        logger.debug("Initialize forms for cavity-constrained mechanics problem")
        u, p, P_lv = dolfin.split(self.state)
        v, q, P_lv_test = dolfin.split(self.state_test)

        F = dolfin.variable(kinematics.DeformationGradient(u))
        J = kinematics.Jacobian(F)
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
        n_weighted = ufl.cofac(F) * N
        cavity_integrand = -dolfin.dot(x, n_weighted) / 3.0

        total_ref_volume = dolfin.assemble(dolfin.Constant(1.0) * dx)

        cavity_piece_1 = -P_lv * cavity_integrand * ds(self.lv_marker)
        cavity_piece_2 = (P_lv * self.v_target / total_ref_volume) * dx
        cavity_energy = cavity_piece_1 + cavity_piece_2
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
            u = get_u_view(self)

            F = dolfin.grad(u) + dolfin.Identity(3)
            f = F * active.f0
            lmbda = dolfin.sqrt(f ** 2)

            active._projector.project(active.lmbda, lmbda)
            if active.dt > 0:
                active._projector.project(
                    active._dLambda,
                    (lmbda - active.lmbda_prev) / active.dt,
                )

            if update_ta_current:
                active._projector.project(active.Ta_current, active.Ta(lmbda))

            active.update_current(lmbda=lmbda)
            active.update_prev()


def _compute_ta_target(cavity_problem):
    active = cavity_problem.material.active
    u = get_u_view(cavity_problem)
    F = dolfin.grad(u) + dolfin.Identity(3)
    f = F * active.f0
    lmbda_expr = dolfin.sqrt(f ** 2)

    scratch = dolfin.Function(active.Ta_current.function_space())
    active._projector.project(scratch, active.Ta(lmbda_expr))
    return scratch.vector().get_local().copy()


def solve_cavity_with_ta_ramp(cavity_problem, max_ta_step=2.0, max_step_doublings=6):
    active = cavity_problem.material.active
    comm = cavity_problem.geometry.mesh.mpi_comm()

    def _ramp_to(target_vals, label):
        Ta_start = active.Ta_current.vector().get_local().copy()
        diff = target_vals - Ta_start
        max_jump = np.abs(diff).max()
        n_steps = max(1, int(np.ceil(max_jump / max_ta_step)))

        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
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
                if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
                    logger.info(f"    [Ta ramp:{label}] substep {i + 1}/{len(alpha_vals)} "
                            f"(alpha={alpha:.3f}): trial_Ta_max={trial_vals.max():.4f}")

                try:
                    nliter, nlconv = cavity_problem._raw_solve()
                    local_ok = 1 if (nlconv and nliter <= 6) else 0
                except RuntimeError as e:
                    logger.info(f"    [Ta ramp:{label}] substep {i + 1} raised RuntimeError: {e}")
                    local_ok = 0
                    nliter, nlconv = None, False

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


# ── LVCycleRunner ────────────────────────────────────────────────────────────

class LVCycleRunner(Runner):
    def set_cycle_controller(self, controller, mech_problem, cavity_problem, outdir):
        self._cycle_controller = controller
        self._mech_problem = mech_problem
        self._cavity_problem = cavity_problem
        self._outdir = outdir
        self._in_cavity_mode = False

        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            self._pv_file = open(os.path.join(outdir, "pv_loop.csv"), "w")
            self._pv_file.write("t_ms,LVP_kPa,LVV,phase,mode\n")
            self._pv_file.flush()
        else:
            self._pv_file = None

    def _solve_mechanics(self):
        n_substeps = 0
        self.coupling.coupling_to_mechanics()

        want_cavity_mode = False  # forced on — minimal repro, skip PRELOAD entirely
        t_ms = TimeStepper.ns2ms(self.t)
        dt_ms = self._config.dt

        if want_cavity_mode and not self._in_cavity_mode:
            if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
                logger.info(f"  [handoff] t={t_ms:.1f}ms entering cavity-constrained mode")
            handoff_uP(self._mech_problem, self._cavity_problem)
            self._in_cavity_mode = True

        state = self._cycle_controller.lv_state
        target_vol = state.volume_n if state.end_dia_vol == 0 else state.end_dia_vol
        self._cavity_problem.v_target.assign(target_vol)

        n_substeps = solve_cavity_with_ta_ramp(self._cavity_problem)
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            logger.info(f"  [cavity solve] Ta ramp used {n_substeps} substep(s) this macro step")

        p_lv_value = get_cavity_pressure(self._cavity_problem)
        u_now = get_u_view(self._cavity_problem)
        v_lv_now = compute_cavity_volume(self._cycle_controller.geometry, u_now,
                                          self._cycle_controller.lv_marker)
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            logger.info(f"  [cavity debug] AFTER solve: P_lv={p_lv_value:.6f}, V_lv={v_lv_now:.1f}")

        commit_step(state, v_lv_now, p_lv_value)
        advance_phase(state, t_ms, dt_ms)

        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            logger.info(f"  [cavity solve] result: P_lv={p_lv_value:.4f} kPa, V_lv={v_lv_now:.1f} mm^3, "
                    f"new_phase={state.phase}")
        self.coupling.interpolate(self.coupling.lmbda_mech, self.coupling.lmbda_ep)

        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            print(f"  → t={t_ms:.1f} ms  phase={state.phase}  LVP={state.pressure_n:.3f} kPa  "
                  f"LVV={state.volume_n:.2f}", flush=True)
            self._pv_file.write(
                f"{t_ms:.3f},{state.pressure_n:.6f},{state.volume_n:.6f},{state.phase},cavity\n"
            )
            self._pv_file.flush()

    def close_files(self):
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0 and self._pv_file:
            self._pv_file.close()


# ── 1. Generate / load LV ellipsoid geometry ────────────────────────────────
GEO_DIR = "/repo/simcardems-dev/demos/geometries"
GEO_PATH = os.path.join(GEO_DIR, "lv_ellipsoid.h5")
GEO_SCHEMA = os.path.join(GEO_DIR, "lv_ellipsoid.json")

if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
    if not os.path.exists(GEO_PATH):
        mpi_print("Generating LV ellipsoid mesh...")
        os.makedirs(GEO_DIR, exist_ok=True)
        cardiac_geometries.create_lv_ellipsoid(
            outdir=GEO_DIR,   # <-- was "."
            r_short_endo=7.0, r_short_epi=10.0,
            r_long_endo=17.0, r_long_epi=20.0,
            psize_ref=4.0,
            fiber_angle_endo=-60.0, fiber_angle_epi=60.0,
            create_fibers=True, fiber_space="Quadrature_3",
        )
# GEO_PATH = "/repo/simcardems-dev/demos/geometries/lv_ellipsoid.h5"
# GEO_SCHEMA = "/repo/simcardems-dev/demos/geometries/lv_ellipsoid.json"

dolfin.MPI.comm_world.barrier()
mpi_print("Loading LV ellipsoid geometry...")
from cardiac_geometries.geometry import Geometry

geo = Geometry.from_file(
    fname=GEO_PATH, schema_path=GEO_SCHEMA,
    schema=LeftVentricularGeometry.default_schema(),
)
lv_geo = LeftVentricularGeometry.from_geometry(geo, ep_mesh=geo.mesh, ffun_ep=geo.ffun)

# ── 2. Whole-mesh uniform activation at end-diastole ────────────────────────

T_END_DIASTOLE = 0.1

mpi_print("Building uniform whole-mesh activation times...")
node_coords = lv_geo.ep_mesh.coordinates()
node_ids = np.arange(len(node_coords))
activation_times = np.full(len(node_coords), T_END_DIASTOLE)

act_fn = interpolate_activation_to_ep_mesh(
    ep_mesh=lv_geo.ep_mesh,
    endo_marker_ep=[lv_geo.markers["ENDO"][0]],
    ffun_ep=lv_geo.ffun_ep,
    node_ids_mech=node_ids,
    activation_times_mech=activation_times,
    coords_subset_mech=node_coords,
)
local_vec = act_fn.vector().get_local()
local_vec[local_vec == np.inf] = T_END_DIASTOLE
act_fn.vector().set_local(local_vec)
act_fn.vector().apply("insert")
dolfin.MPI.comm_world.barrier()

mpi_print("Building whole-mesh stimulus domain...")
tdim = lv_geo.ep_mesh.topology().dim()
cell_domain = dolfin.MeshFunction("size_t", lv_geo.ep_mesh, tdim)
cell_domain.set_all(1)
lv_geo.stimulus_domain = StimulusDomain(domain=cell_domain, marker=1)

# ── 3. Config ────────────────────────────────────────────────────────────

mpi_print("Configuring...")
config = Config()
config.T = 100 # just a couple macro steps — we only need the ramp to fire a few times
config.dt = 0.5
config.dt_mech = 1.0
config.mech_threshold = 1.0
config.geometry_path = GEO_PATH
config.outdir = "output_lv_ellipsoid_isovol_lagrange"
config.coupling_type = "fully_coupled_Tor_Land"
config.save_freq = 1
config.linear_mechanics_solver = "mumps"
config.spring = 10.0
config.debug_mode = True
config.mechanics_use_custom_newton_solver = False

# ── 4. Build EM coupling ─────────────────────────────────────────────────

mpi_print("Setting up EM model...")
coupling = em_model.setup_EM_model_from_config(
    config, geometry=lv_geo, activation_times=act_fn,
)

# ── 5. LV-only cycle controller + cavity-constrained problem ──────────────

LV_ENDO_MARKER = lv_geo.markers["ENDO"][0]
mech_problem = coupling.mech_solver

lv_params = CycleParams(
    t_zero=5.0, t_prestress=0.0, preload_pressure=0.05, prestress_pressure=0.0,
    t_end_diastole=T_END_DIASTOLE, p_end_diastole=1.0,
    gain_contraction=(1.0, 0.5), gain_relaxation=(0.5, 0.2),
    p_fill=0.1, period=800.0,
    windkessel=WindkesselParams(p_init=0.3, compliance=1.0, resistance=100.0, evolve=True),
)
rv_params = CycleParams(
    t_zero=1e9, t_prestress=0.0, preload_pressure=0.0, prestress_pressure=0.0,
    t_end_diastole=1e9, p_end_diastole=0.0,
    gain_contraction=(1.0, 0.5), gain_relaxation=(0.5, 0.2),
    p_fill=0.0, period=800.0,
    windkessel=WindkesselParams(p_init=0.0, compliance=1.0, resistance=1.0, evolve=False),
)

lv_state = CavityState(name="LV", params=lv_params)
rv_state = CavityState(name="RV_stub", params=rv_params)

lv_pressure_const = None
for nbc in mech_problem.bcs.neumann:
    if nbc.marker == LV_ENDO_MARKER:
        lv_pressure_const = nbc.traction
        break
if lv_pressure_const is None:
    raise RuntimeError("No Neumann BC found on ENDO")
rv_pressure_const = dolfin.Constant(0.0)

cycle_controller = BiVCycleController(
    lv_state=lv_state, rv_state=rv_state,
    lv_pressure_constant=lv_pressure_const, rv_pressure_constant=rv_pressure_const,
    geometry=lv_geo, lv_marker=LV_ENDO_MARKER, rv_marker=LV_ENDO_MARKER,
)

u0, _ = mech_problem.state.split(deepcopy=True)  # one-time, safe at low rank / init only
cycle_controller.initialize(u0)
mpi_print(f"Initial LV volume: {lv_state.volume_n:.2f}")
lv_state.end_dia_vol = lv_state.volume_n

mpi_print("Setting up cavity-constrained mechanics problem...")
robin_bcs = list(mech_problem.bcs.robin)
cavity_bcs = pulse.BoundaryConditions(robin=robin_bcs, dirichlet=[], neumann=[])
v_target_lv = dolfin.Constant(float(lv_state.volume_n))

cavity_problem = CavityConstrainedMechanicsProblem(
    lv_geo, mech_problem.material, cavity_bcs,
    lv_marker=LV_ENDO_MARKER, v_target=v_target_lv,
)

# ── 6. Time loop ──────────────────────────────────────────────────────────

mpi_print("Starting time loop...")
os.makedirs(config.outdir, exist_ok=True)

runner = LVCycleRunner.from_models(coupling=coupling, config=config)
runner.set_cycle_controller(cycle_controller, mech_problem, cavity_problem, config.outdir)

try:
    runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=False)
except Exception as e:
    mpi_print(f"Runner crashed: {e}")
finally:
    runner.close_files()
    mpi_print("Done.")