"""
run_lv_ellipsoid_isovol_lagrange.py
------------------------------------
LV ellipsoid cardiac cycle test — Lagrange-multiplier (volume-constrained)
mechanics ONLY during isovolumic phases (IVC/IVR). PRELOAD/EJECTION/FILLING
still use the standard pressure-driven Neumann BC + pulse.iterate, exactly
as before.

Goal of this specific test: confirm (1) the cavity-constrained solve
converges in isovolumic phases without the old secant/fixed-point loop,
and (2) phase transitions (PRELOAD -> IVC -> EJECTION -> IVR -> FILLING)
still fire correctly with this hybrid approach.
"""

import os
cache_dir = os.environ.get("FENICS_CACHE_DIR", os.path.expanduser("~/.cache"))
os.environ["XDG_CACHE_HOME"] = cache_dir

import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import dolfin
dolfin.set_log_level(dolfin.LogLevel.DEBUG)

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
from simcardems.activation import interpolate_activation_to_ep_mesh, endocardial_stimulus_domain
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
)

logger = logging.getLogger(__name__)


def mpi_print(*args, **kwargs):
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print(*args, **kwargs, flush=True)


# ── CavityConstrainedMechanicsProblem ───────────────────────────────────────
# Equations 4-6 of Mehri et al. 2025 (MyoFE), minus rigid-body multipliers,
# minus base Dirichlet BC — rigid-body motion still handled by whatever
# BCs are passed in (here: the same Robin EPI spring as the main problem).

class CavityConstrainedMechanicsProblem(pulse.MechanicsProblem):
    def __init__(self, *args, lv_marker=None, v_target=None, **kwargs):
        if lv_marker is None:
            raise ValueError("Must supply lv_marker (the ENDO facet marker int)")
        self.lv_marker = lv_marker
        self.v_target = v_target if v_target is not None else dolfin.Constant(0.0)
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

        # ── Real active-stress coupling, matching simcardems's own pattern
        #    verbatim (mechanics_model.py) — strain_energy(F)'s built-in
        #    Wactive is a dead path for LandModel (uses self._activation,
        #    which LandModel never sets); the actual live coupling is this
        #    explicit "frozen Ta_current" term, added directly here. ──────
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
        return super().solve()  # exactly pulse.MechanicsProblem.solve(), unchanged

    def _update_active_stress_bookkeeping(self, update_ta_current=True):
        if self.strong_coupling:
            active = self.material.active
            u, p, P_lv = self.state.split(deepcopy=True)

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

def handoff_uP(source_problem, dest_problem):
    """Copy (u, p) from source_problem.state into dest_problem.state's
    corresponding sub-blocks — works regardless of whether either side
    has an extra P_lv block, since we only ever touch sub(0)/sub(1)."""
    parts = source_problem.state.split(deepcopy=True)
    u_src, p_src = parts[0], parts[1]
    dolfin.assign(dest_problem.state.sub(0), u_src)
    dolfin.assign(dest_problem.state.sub(1), p_src)

def _compute_ta_target(cavity_problem):
    """What Ta *would* be right now, using the last-solved displacement
    and the freshly-interpolated XS_mech/XW_mech — the actual quantity
    we want to ramp Ta_current TOWARD. Computed via a scratch Function
    so it doesn't touch Ta_current itself."""
    active = cavity_problem.material.active
    u, p, P_lv = cavity_problem.state.split(deepcopy=True)
    F = dolfin.grad(u) + dolfin.Identity(3)
    f = F * active.f0
    lmbda_expr = dolfin.sqrt(f ** 2)

    scratch = dolfin.Function(active.Ta_current.function_space())
    active._projector.project(scratch, active.Ta(lmbda_expr))
    return scratch.vector().get_local().copy()


def solve_cavity_with_ta_ramp(cavity_problem, max_ta_step=2.0, max_step_doublings=6):
    """
    Ramp Ta_current from its last committed value toward the EP-driven
    target, then do a final consistency correction: recompute Ta from
    the lambda the ramp actually produced, and solve once more so the
    final state has Ta and lambda mutually consistent (no residual lag
    from computing the ramp's target off a slightly-stale lambda).
    """
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

                logger.info(f"    [Ta ramp:{label}] substep {i + 1}/{len(alpha_vals)} "
                            f"(alpha={alpha:.3f}): trial_Ta_max={trial_vals.max():.4f}")

                try:
                    nliter, nlconv = cavity_problem._raw_solve()
                    if not nlconv or nliter > 6:
                        ok = False
                        logger.info(f"    [Ta ramp:{label}] substep {i + 1} failed/marginal "
                                    f"(nlconv={nlconv}, nliter={nliter})")
                        break
                    logger.info(f"    [Ta ramp:{label}] substep {i + 1} converged in {nliter} iters")
                except RuntimeError:
                    ok = False
                    logger.info(f"    [Ta ramp:{label}] substep {i + 1} raised RuntimeError")
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

    # ── Pass 1: ramp toward the EP-driven target (from pre-ramp lambda) ──
    Ta_target = _compute_ta_target(cavity_problem)
    n_steps_main = _ramp_to(Ta_target, "main")

    # ── Pass 2: consistency correction using the lambda the ramp actually
    #    produced — should be a small delta if the ramp converged cleanly,
    #    since lambda barely moves once Ta itself is near-converged. ─────
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
    def set_cycle_controller(self, controller, mech_problem, cavity_problem,
                              outdir, pseudo_ecg=None, dt_ecg=5.0):
        self._cycle_controller = controller
        self._mech_problem = mech_problem
        self._cavity_problem = cavity_problem
        self._outdir = outdir
        self._pseudo_ecg = pseudo_ecg
        self._dt_ecg = dt_ecg
        self._last_ecg_t = -dt_ecg
        self._in_cavity_mode = False  # tracks which problem currently "owns" state

        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            self._pv_file = open(os.path.join(outdir, "pv_loop.csv"), "w")
            self._pv_file.write("t_ms,LVP_kPa,LVV,phase,mode\n")
            self._pv_file.flush()
            if pseudo_ecg is not None:
                self._ecg_file = open(os.path.join(outdir, "pseudo_ecg.csv"), "w")
                lead_names = ",".join(pseudo_ecg.leads.keys())
                self._ecg_file.write(f"t_ms,{lead_names}\n")
                self._ecg_file.flush()
            else:
                self._ecg_file = None
        else:
            self._pv_file = None
            self._ecg_file = None

    def _solve_mechanics(self):
        n_substeps = 0
        self.coupling.coupling_to_mechanics()

        isovol_phases = (Phase.ISOVOL_CONTRACTION, Phase.ISOVOL_RELAXATION)
        phase = self._cycle_controller.lv_state.phase
        want_cavity_mode = phase in isovol_phases

        t_ms = TimeStepper.ns2ms(self.t)
        dt_ms = self._config.dt

        if want_cavity_mode and not self._in_cavity_mode:
            # ── Entering isovolumic phase: hand state OVER to cavity_problem ──
            logger.info(f"  [handoff] t={t_ms:.1f}ms entering cavity-constrained mode (phase={phase})")
            handoff_uP(self._mech_problem, self._cavity_problem)
            self._in_cavity_mode = True

        elif not want_cavity_mode and self._in_cavity_mode:
            handoff_uP(self._cavity_problem, self._mech_problem)
            self._in_cavity_mode = False

            # Check: does the handoff itself preserve volume, before any new solve?
            u_check, _ = self._mech_problem.state.split(deepcopy=True)
            v_check = compute_cavity_volume(self._cycle_controller.geometry, u_check,
                                            self._cycle_controller.lv_marker)
            logger.info(f"  [handoff check] volume immediately after handoff (pre-solve): {v_check:.1f} mm^3")

            _, _, P_lv_final = self._cavity_problem.state.split(deepcopy=True)
            p_final = P_lv_final.vector().get_local()[0]
            self._cycle_controller.lv_pressure_constant.assign(p_final)

        if self._in_cavity_mode:
            state = self._cycle_controller.lv_state
            target_vol = (state.end_dia_vol if phase == Phase.ISOVOL_CONTRACTION
                          else state.end_sys_vol)
            self._cavity_problem.v_target.assign(target_vol)

            xs_before = self._cavity_problem.material.active.XS.vector().max()
            ta_before = self._cavity_problem.material.active.Ta_current.vector().max()
            lmbda_before = self._cavity_problem.material.active.lmbda.vector().max()

            logger.info(f"  [cavity debug] BEFORE solve: XS_max={xs_before:.6e}, "
                        f"Ta_current_max={ta_before:.6e}, lmbda_max={lmbda_before:.6f}, "
                        f"target_vol={target_vol:.1f}")

            n_substeps = solve_cavity_with_ta_ramp(self._cavity_problem)
            logger.info(f"  [cavity solve] Ta ramp used {n_substeps} substep(s) this macro step")

            _, _, P_lv_fn = self._cavity_problem.state.split(deepcopy=True)
            p_lv_value = P_lv_fn.vector().get_local()[0]

            xs_after = self._cavity_problem.material.active.XS.vector().max()
            ta_after = self._cavity_problem.material.active.Ta_current.vector().max()
            lmbda_after = self._cavity_problem.material.active.lmbda.vector().max()

            u_now, _, _ = self._cavity_problem.state.split(deepcopy=True)
            v_lv_now = compute_cavity_volume(self._cycle_controller.geometry, u_now,
                                             self._cycle_controller.lv_marker)

            logger.info(f"  [cavity debug] AFTER solve: P_lv={p_lv_value:.6f}, V_lv={v_lv_now:.1f}, "
                        f"XS_max={xs_after:.6e}, Ta_current_max={ta_after:.6e}, lmbda_max={lmbda_after:.6f}")

            from simcardems.biv_cavity_cycle_controller import commit_step, advance_phase
            commit_step(state, v_lv_now, p_lv_value)
            logger.info(f"  [wdk debug] BEFORE advance_phase: pressure_n={state.pressure_n:.4f}, "
                        f"wdk_pressure_n={state.wdk_pressure_n:.4f}")
            advance_phase(state, t_ms, dt_ms)

            logger.info(f"  [wdk debug] AFTER advance_phase: new_phase={state.phase}, "
                        f"wdk_pressure_n={state.wdk_pressure_n:.4f}")

            logger.info(f"  [cavity solve] result: P_lv={p_lv_value:.4f} kPa, V_lv={v_lv_now:.1f} mm^3, "
                        f"new_phase={state.phase}")
            self.coupling.interpolate(self.coupling.lmbda_mech, self.coupling.lmbda_ep)
        else:
            self.coupling.solve_mechanics()
            self.coupling.update_prev_mechanics()
            self.coupling.mechanics_to_coupling()
            self.coupling.coupling_to_ep()
            self._cycle_controller.step(problem=self._mech_problem, t=t_ms, dt=dt_ms)

        state = self._cycle_controller.lv_state
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            from tqdm import tqdm
            ta_max_now = self._cavity_problem.material.active.Ta_current.vector().max()
            tqdm.write(
                f"  → t={t_ms:.1f} ms  phase={state.phase}  mode={'cavity' if self._in_cavity_mode else 'pressure'}"
                f"  LVP={state.pressure_n:.3f} kPa  LVV={state.volume_n:.2f}  Ta_max={ta_max_now:.4f} kPa"
            )
            self._pv_file.write(
                f"{t_ms:.3f},{state.pressure_n:.6f},{state.volume_n:.6f},"
                f"{state.phase},{'cavity' if self._in_cavity_mode else 'pressure'},{n_substeps}\n"
            )
            self._pv_file.flush()

        if self._pseudo_ecg is not None and t_ms - self._last_ecg_t >= self._dt_ecg - 1e-10:
            self.coupling.assigners.assign_ep()
            v_fn = self.coupling.assigners.functions["ep"]["V"]
            self._pseudo_ecg.compute(v_fn, t_ms)
            self._last_ecg_t = t_ms
            if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0 and self._ecg_file is not None:
                vals = ",".join(f"{v:.6f}" for v in self._pseudo_ecg.last_values)
                self._ecg_file.write(f"{t_ms:.3f},{vals}\n")
                self._ecg_file.flush()

    def close_files(self):
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            if self._pv_file:
                self._pv_file.close()
            if self._ecg_file:
                self._ecg_file.close()


# ── 1. Generate / load LV ellipsoid geometry (unchanged) ───────────────────

GEO_PATH   = "/repo/simcardems-dev/demos/geometries/lv_ellipsoid.h5"
GEO_SCHEMA = "/repo/simcardems-dev/demos/geometries/lv_ellipsoid.json"

if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
    if not os.path.exists(GEO_PATH):
        mpi_print("Generating LV ellipsoid mesh...")
        cardiac_geometries.create_lv_ellipsoid(
            outdir=".",
            r_short_endo=7.0, r_short_epi=10.0,
            r_long_endo=17.0, r_long_epi=20.0,
            psize_ref=4.0,
            fiber_angle_endo=-60.0, fiber_angle_epi=60.0,
            create_fibers=True, fiber_space="Quadrature_3",
        )

dolfin.MPI.comm_world.barrier()
mpi_print("Loading LV ellipsoid geometry...")
from cardiac_geometries.geometry import Geometry

geo = Geometry.from_file(
    fname=GEO_PATH, schema_path=GEO_SCHEMA,
    schema=LeftVentricularGeometry.default_schema(),
)
lv_geo = LeftVentricularGeometry.from_geometry(geo, ep_mesh=geo.mesh, ffun_ep=geo.ffun)

from simcardems.postprocess import ecg_recovery


class PseudoECG:
    def __init__(self, leads: dict, sigma_b: float = 1.0):
        self.leads = leads
        self.sigma_b = sigma_b
        self.log = {name: [] for name in leads}
        self.times = []

    def compute(self, v: dolfin.Function, t: float):
        mesh = v.function_space().mesh()
        self.last_values = []
        for name, (pos, neg) in self.leads.items():
            phi_pos = ecg_recovery(v=v, sigma_b=self.sigma_b, point=np.array(pos), mesh=mesh)
            phi_neg = ecg_recovery(v=v, sigma_b=self.sigma_b, point=np.array(neg), mesh=mesh)
            val = phi_pos - phi_neg
            self.log[name].append(val)
            self.last_values.append(val)
        self.times.append(t)


pseudo_ecg = PseudoECG(
    leads={
        "I":   ((100,   0,  50), (-100,   0,  50)),
        "II":  (( 50, -100, 50), ( -50, 100,  50)),
        "III": (( 50, -100, 50), ( 100,   0,  50)),
        "aVR": ((-100,   0, 50), (  75, -50,  50)),
        "aVL": (( 100,   0, 50), ( -25, -50,  50)),
        "aVF": ((   0, -100, 50), (   0,  50,  50)),
        "V1":  ((  20,  -30, 30), (  0,   0,  50)),
        "V2":  ((  10,  -35, 20), (  0,   0,  50)),
        "V3":  (( -10,  -35, 10), (  0,   0,  50)),
        "V4":  (( -20,  -30,  0), (  0,   0,  50)),
        "V5":  (( -30,  -20,-10), (  0,   0,  50)),
        "V6":  (( -35,    0,-10), (  0,   0,  50)),
    },
    sigma_b=1.0,
)

# ── 2. Whole-mesh uniform activation at end-diastole (unchanged) ───────────

T_END_DIASTOLE = 10.0

mpi_print("Building uniform whole-mesh activation times...")
node_coords = lv_geo.ep_mesh.coordinates()
node_ids    = np.arange(len(node_coords))
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

# ── 3. Config (unchanged) ────────────────────────────────────────────────

mpi_print("Configuring...")
config = Config()
config.T = 800.0
config.dt = 0.05
config.mechanics_solve_strategy = "hybrid"
config.dt_mech = 0.05
config.mech_threshold = 1.0
config.geometry_path = GEO_PATH
config.outdir = "output_lv_ellipsoid_isovol_lagrange"
config.coupling_type = "fully_coupled_Tor_Land"
config.save_freq = 5
config.linear_mechanics_solver = "mumps"
config.spring = 10.0
config.debug_mode = True
config.mechanics_use_custom_newton_solver = False

# ── 4. Build EM coupling (unchanged) ────────────────────────────────────

mpi_print("Setting up EM model...")
coupling = em_model.setup_EM_model_from_config(
    config, geometry=lv_geo, activation_times=act_fn,
)

# ── 5. LV-only cycle controller (unchanged) + cavity-constrained problem ──

LV_ENDO_MARKER = lv_geo.markers["ENDO"][0]
mech_problem = coupling.mech_solver

lv_pressure_const = None
for nbc in mech_problem.bcs.neumann:
    if nbc.marker == LV_ENDO_MARKER:
        lv_pressure_const = nbc.traction
        break
if lv_pressure_const is None:
    raise RuntimeError("No Neumann BC found on ENDO")

rv_pressure_const = dolfin.Constant(0.0)

lv_params = CycleParams(
    t_zero=5.0, t_prestress=0.0, preload_pressure=0.05, prestress_pressure=0.0,
    t_end_diastole=T_END_DIASTOLE, p_end_diastole=0.1,
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

cycle_controller = BiVCycleController(
    lv_state=lv_state, rv_state=rv_state,
    lv_pressure_constant=lv_pressure_const, rv_pressure_constant=rv_pressure_const,
    geometry=lv_geo, lv_marker=LV_ENDO_MARKER, rv_marker=LV_ENDO_MARKER,
)

u0, _ = mech_problem.state.split(deepcopy=True)
cycle_controller.initialize(u0)
mpi_print(f"Initial LV volume: {lv_state.volume_n:.2f}")

# ── Build the cavity-constrained problem, sharing material + Robin BCs ────
mpi_print("Setting up cavity-constrained mechanics problem...")
robin_bcs = list(mech_problem.bcs.robin)  # reuse exactly what's already there
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
runner.set_cycle_controller(cycle_controller, mech_problem, cavity_problem,
                             config.outdir, pseudo_ecg=pseudo_ecg, dt_ecg=5.0)

try:
    runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=True)
except Exception as e:
    mpi_print(f"Runner crashed: {e}")
finally:
    runner.close_files()
    mpi_print("Done.")