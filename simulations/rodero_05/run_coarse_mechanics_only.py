import os

cache_dir = os.environ.get("FENICS_CACHE_DIR", os.path.expanduser("~/.cache"))
os.environ["XDG_CACHE_HOME"] = cache_dir

import logging
logging.getLogger("simcardems.newton_solver").setLevel(logging.DEBUG)
logging.getLogger("simcardems.biv_cavity_cycle_controller").setLevel(logging.INFO)
logging.getLogger("__main__").setLevel(logging.DEBUG)
import dataclasses
import dolfin
dolfin.PETScOptions.set("mat_mumps_icntl_4", "0")
import numpy as np
import pandas as pd
import pulse
from simcardems.config import Config
from simcardems import mechanics_model
from simcardems.time_stepper import TimeStepper
from simcardems.biv_cavity_cycle_controller import (
    BiVCycleController,
    CavityState,
    CycleParams,
    WindkesselParams,
    Phase,
)
from simcardems.geometry import refine_mesh
from simcardems import utils

import time
t_script_start = time.time()

logger = utils.getLogger(__name__)
logger.info(['dolfin:', dolfin.__version__])
import petsc4py; logger.info(['petsc4py:', petsc4py.__version__])
from petsc4py import PETSc; logger.info(['PETSc:', PETSc.Sys.getVersion()])

RESOLUTION = "4mm"
MESH_DIR = "meshes/"
RESULTS_DIR = f"results_mech_only_{RESOLUTION}/"

WARM_START_T_MS = None  # mechanics-only run — no EP state to warm-start from

# ── Phase-dependent dt/dt_mech (unchanged from run_coarse.py) ─────────────
def dt_targets_for_phase(phase):
    if phase == Phase.PRELOAD:
        return dict(dt=10.0, dt_mech=10.0)
    else:
        return dict(dt=0.5, dt_mech=1.0)


def apply_phase_dt(config, phase, logger=None):
    targets = dt_targets_for_phase(phase)
    changed = (config.dt != targets["dt"]) or (config.dt_mech != targets["dt_mech"])
    if changed and logger is not None:
        logger.info(f"dt update (phase={phase}): "
                    f"dt {config.dt} -> {targets['dt']}, "
                    f"dt_mech {config.dt_mech} -> {targets['dt_mech']}")
    config.dt = targets["dt"]
    config.dt_mech = targets["dt_mech"]
    return changed


# ── Isovolumic secant convergence (unchanged from run_coarse.py) ─────────
def converge_isovolumic_pressure(solve_mechanics_fn, biv_geo,
                                   lv_pressure_const, rv_pressure_const,
                                   target_vol_lv, target_vol_rv,
                                   mech_problem, max_iters=3, tol_vol=300.0):
    def get_volumes():
        u_now, _ = mech_problem.state.split(deepcopy=True)
        v_lv = pulse.dolfin_utils.get_cavity_volume(biv_geo, chamber="lv", u=u_now)
        v_rv = pulse.dolfin_utils.get_cavity_volume(biv_geo, chamber="rv", u=u_now)
        return v_lv, v_rv

    p_lv, p_rv = float(lv_pressure_const), float(rv_pressure_const)

    logger.info("    [isovol converge] mandatory first solve at current pressure")
    solve_mechanics_fn()
    v_lv, v_rv = get_volumes()
    err_lv, err_rv = v_lv - target_vol_lv, v_rv - target_vol_rv

    logger.info(f"    [isovol converge] target: LV={target_vol_lv:.1f}mm^3 RV={target_vol_rv:.1f}mm^3")
    logger.info(f"    [isovol converge] after mandatory solve: p_lv={p_lv:.4f} v_lv={v_lv:.1f} err_lv={err_lv:+.1f}  |  "
                f"p_rv={p_rv:.4f} v_rv={v_rv:.1f} err_rv={err_rv:+.1f}")

    p_lv_prev, err_lv_prev = p_lv, err_lv
    p_rv_prev, err_rv_prev = p_rv, err_rv

    if abs(err_lv) < tol_vol and abs(err_rv) < tol_vol:
        logger.info("    [isovol converge] CONVERGED immediately (0 corrections needed after mandatory solve)")
        return True, 0

    for it in range(max_iters):
        if it == 0:
            dp_lv = max(0.01, 0.05 * abs(p_lv))
            dp_rv = max(0.01, 0.05 * abs(p_rv))
            p_lv_trial = p_lv + np.sign(err_lv) * dp_lv if err_lv != 0 else p_lv
            p_rv_trial = p_rv + np.sign(err_rv) * dp_rv if err_rv != 0 else p_rv
            logger.info(f"    [isovol converge] iter {it}: no slope history yet, "
                        f"perturbing p_lv {p_lv:.4f}->{p_lv_trial:.4f}, "
                        f"p_rv {p_rv:.4f}->{p_rv_trial:.4f}")
        else:
            slope_lv = (err_lv - err_lv_prev) / (p_lv - p_lv_prev) if p_lv != p_lv_prev else None
            slope_rv = (err_rv - err_rv_prev) / (p_rv - p_rv_prev) if p_rv != p_rv_prev else None
            p_lv_trial = (p_lv - err_lv / slope_lv) if slope_lv and abs(slope_lv) > 1e-9 else p_lv
            p_rv_trial = (p_rv - err_rv / slope_rv) if slope_rv and abs(slope_rv) > 1e-9 else p_rv
            logger.info(f"    [isovol converge] iter {it}: slope_lv={slope_lv}, slope_rv={slope_rv}")
            logger.info(f"    [isovol converge] iter {it}: secant update "
                        f"p_lv {p_lv:.4f}->{p_lv_trial:.4f}, p_rv {p_rv:.4f}->{p_rv_trial:.4f}")

        lv_pressure_const.assign(p_lv_trial)
        rv_pressure_const.assign(p_rv_trial)
        solve_mechanics_fn()

        p_lv_prev, err_lv_prev = p_lv, err_lv
        p_rv_prev, err_rv_prev = p_rv, err_rv
        p_lv, p_rv = p_lv_trial, p_rv_trial

        v_lv, v_rv = get_volumes()
        err_lv, err_rv = v_lv - target_vol_lv, v_rv - target_vol_rv
        logger.info(f"    [isovol converge] iter {it} result: "
                    f"v_lv={v_lv:.1f} err_lv={err_lv:+.1f}  |  v_rv={v_rv:.1f} err_rv={err_rv:+.1f}")

        if abs(err_lv) < tol_vol and abs(err_rv) < tol_vol:
            logger.info(f"    [isovol converge] CONVERGED after {it + 1} correction(s)")
            return True, it + 1

    converged = abs(err_lv) < tol_vol and abs(err_rv) < tol_vol
    logger.info(f"    [isovol converge] {'CONVERGED' if converged else 'DID NOT CONVERGE'} "
                f"after max_iters={max_iters} "
                f"(final |err_lv|={abs(err_lv):.1f}, |err_rv|={abs(err_rv):.1f}, tol={tol_vol})")
    return converged, max_iters


def check_orthonormality(f0, s0, n0, label=""):
    def as_array(x):
        if hasattr(x, "vector"):
            return x.vector().get_local().reshape(-1, 3)
        return x
    F, S, N = as_array(f0), as_array(s0), as_array(n0)
    f0n = np.linalg.norm(F, axis=1)
    dot_fs = np.abs(np.sum(F * S, axis=1))
    dot_fn = np.abs(np.sum(F * N, axis=1))
    dot_sn = np.abs(np.sum(S * N, axis=1))
    logger.info(f"{label} f0 norm: {f0n.min():.6f}-{f0n.max():.6f}, "
                f"max|f0.s0|={dot_fs.max():.3e}, max|f0.n0|={dot_fn.max():.3e}, max|s0.n0|={dot_sn.max():.3e}")


# ── 1. Geometry (unchanged) ────────────────────────────────────────────────
logger.info('Build geometry...')
from cardiac_geometries.geometry import Geometry
from simcardems.bivgeometry import BiVentricularGeometry
from simcardems.spatial_fields import map_dense_field_to_ep_mesh

geo = Geometry.from_file(MESH_DIR + "rodero_05_coarse_" + RESOLUTION + ".h5")

mesh = dolfin.Mesh()
with dolfin.HDF5File(mesh.mpi_comm(), MESH_DIR + "rodero_05_coarse_" + RESOLUTION + ".h5", "r") as f:
    f.read(mesh, "mesh", False)

coords = mesh.coordinates()
cells_arr = mesh.cells()
logger.info(f"Mesh: {mesh.num_vertices()} vertices, {mesh.num_cells()} cells")

biv_geo = BiVentricularGeometry.from_geometry(geo, ep_mesh=geo.mesh, ffun_ep=geo.ffun)
logger.info(f"Mechanics Mesh vertices: {biv_geo.mechanics_mesh.num_vertices()}")

# Valve plug mask (unchanged)
coarse_tv = np.load(MESH_DIR + '/rodero_05_coarse_' + RESOLUTION + '_tv.npy')
is_valve_float = (coarse_tv >= 7).astype(float)
coarse_centres_all = np.array([cell.midpoint().array()
                                for cell in dolfin.cells(biv_geo.mechanics_mesh)])
valve_fn = map_dense_field_to_ep_mesh(biv_geo.mechanics_mesh, coarse_centres_all, is_valve_float)
logger.debug(f'Mechanics valve plug elements: {int(is_valve_float.sum())}')
biv_geo.valve_mask = valve_fn

check_orthonormality(biv_geo.f0, biv_geo.s0, biv_geo.n0)

# ── 2. Config ────────────────────────────────────────────────────────────
logger.info('Configuring...')
config = Config()
config.T = 120.0   # mechanics-only: no need to run a full 800ms beat
config.dt = 10.0
config.dt_mech = 10.0
config.outdir = RESULTS_DIR + "biv_coarse_mech_only_output"
config.save_freq = 10
config.linear_mechanics_solver = "mumps"
config.spring = 50.0
config.traction = 0.0
config.mechanics_use_custom_newton_solver = True
config.mechanics_solve_strategy = "fixed"
config.relaxation_factor = 1.0

material_params_override = dict(
    a=2.28, a_f=1.686, b=9.726, b_f=15.779,
    a_s=0.0, b_s=0.0, a_fs=0.0, b_fs=0.0,
)

VALVE_STIFFNESS_SCALE = 3.0
logger.info("=" * 60)
logger.info("RUN PARAMETERS (mechanics-only)")
logger.info("=" * 60)
logger.info(f"Resolution:                {RESOLUTION}")
logger.info(f"T (total time):            {config.T} ms")
logger.info(f"Linear mechanics solver:   {config.linear_mechanics_solver}")
logger.info(f"Custom Newton solver:      {config.mechanics_use_custom_newton_solver}")
logger.info(f"Spring (EPI Robin):        {config.spring}")
logger.info(f"Valve stiffness scale:     {VALVE_STIFFNESS_SCALE}")
logger.info("Material parameters:")
for k, v in material_params_override.items():
    logger.info(f"  {k:6s} = {v}")
logger.info(f"Output directory:          {config.outdir}")
logger.info("=" * 60)

os.makedirs(config.outdir, exist_ok=True)

# ── 3. Monkey-patch: normal-only Robin BC (identical to run_coarse.py) ───
import pulse.mechanicsproblem as _mp
from pulse import kinematics as _kinematics
from pulse.dolfin_utils import list_sum as _list_sum
try:
    import ufl_legacy as _ufl
except ImportError:
    import ufl as _ufl

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
        external_work.append(dolfin.inner(robin.value * u_normal, v) * ds(robin.marker))
    for body_force in self.bcs.body_force:
        external_work.append(-dolfin.derivative(dolfin.inner(body_force, u) * dx, u, v))
    return _list_sum(external_work) if external_work else None

_mp.MechanicsProblem._external_work = _external_work_normal_robin
logger.info("Patched MechanicsProblem._external_work: Robin BC normal-only")

# ── 4. Material + mechanics problem — via the REAL create_problem(), ────
#     no active model (no EP), same BCs/Newton-solver/valve-scaling code
#     path as the full run_coarse.py ────────────────────────────────────
class PassiveActiveModel(pulse.ActiveModel):
    """Zero-activation active model with a no-op update_prev, satisfying
    simcardems's custom Newton solver, which expects
    material.active.update_prev to exist as its per-step update callback."""
    def update_prev(self):
        pass

material = pulse.HolzapfelOgden(
    active_model=PassiveActiveModel(f0=biv_geo.f0, s0=biv_geo.s0, n0=biv_geo.n0),
    parameters=material_params_override,
)

logger.info("Starting MechanicsProblem init (via mechanics_model.create_problem)...")
mech_problem = mechanics_model.create_problem(
    material=material,
    geo=biv_geo,
    bnd_rigid=config.bnd_rigid,
    spring=config.spring,
    traction=config.traction,
    fix_right_plane=config.fix_right_plane,
    linear_solver="gmres",
    use_custom_newton_solver=config.mechanics_use_custom_newton_solver,
    debug_mode=config.debug_mode,
    base_displacement=None,
    valve_stiffness_scale=VALVE_STIFFNESS_SCALE,
)
logger.info("MechanicsProblem init done. Starting initial solve...")
mech_problem.solve()
logger.info("Initial mechanics solve done.")

# ── 5. Cycle controller (unchanged) ──────────────────────────────────────
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
logger.debug(f"Initial LV volume: {lv_state.volume_n:.2f}")
logger.debug(f"Initial RV volume: {rv_state.volume_n:.2f}")

# ── 6. Time loop — mechanics + cycle controller only, no EP stepping ────
apply_phase_dt(config, cycle_controller.lv_state.phase, logger=logger)

if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
    pv_path = os.path.join(config.outdir, "pv_loop.csv")
    pv_file = open(pv_path, "w")
    pv_file.write("t_ms,LVP_kPa,LVV,RVP_kPa,RVV,LV_phase,RV_phase\n")
else:
    pv_file = None

xdmf = dolfin.XDMFFile(os.path.join(config.outdir, "mechanics_only.xdmf"))

t = 0.0
try:
    while t < config.T:
        t += config.dt
        logger.debug(f"Solve mechanics at t={t:.2f} ms")

        phase = cycle_controller.lv_state.phase
        if phase == Phase.ISOVOL_CONTRACTION:
            target_vol_lv = cycle_controller.lv_state.end_dia_vol
            target_vol_rv = cycle_controller.rv_state.end_dia_vol
            logger.info("  [isovol] phase=ISOVOL_CONTRACTION, targeting end-diastolic volumes")
            converged, n_it = converge_isovolumic_pressure(
                mech_problem.solve, biv_geo, lv_pressure_const, rv_pressure_const,
                target_vol_lv, target_vol_rv, mech_problem,
            )
            logger.info(f"  Isovolumic pressure convergence: converged={converged} in {n_it} iters")
        elif phase == Phase.ISOVOL_RELAXATION:
            target_vol_lv = cycle_controller.lv_state.end_sys_vol
            target_vol_rv = cycle_controller.rv_state.end_sys_vol
            logger.info("  [isovol] phase=ISOVOL_RELAXATION, targeting end-systolic volumes")
            converged, n_it = converge_isovolumic_pressure(
                mech_problem.solve, biv_geo, lv_pressure_const, rv_pressure_const,
                target_vol_lv, target_vol_rv, mech_problem,
            )
            logger.info(f"  Isovolumic pressure convergence: converged={converged} in {n_it} iters")
        else:
            mech_problem.solve()

        cycle_controller.step(problem=mech_problem, t=t, dt=config.dt)
        apply_phase_dt(config, cycle_controller.lv_state.phase, logger=logger)

        lv, rv = cycle_controller.lv_state, cycle_controller.rv_state
        logger.debug(
            f"  → t={t:.1f} ms  LV phase={lv.phase}  LVP={lv.pressure_n:.3f} kPa  LVV={lv.volume_n/1000:.2f} mL"
            f"  RV phase={rv.phase}  RVP={rv.pressure_n:.3f} kPa  RVV={rv.volume_n/1000:.2f} mL"
        )
        if pv_file is not None:
            pv_file.write(f"{t:.3f},{lv.pressure_n:.6f},{lv.volume_n:.6f},"
                           f"{rv.pressure_n:.6f},{rv.volume_n:.6f},{lv.phase},{rv.phase}\n")
            pv_file.flush()

        u, _ = mech_problem.state.split(deepcopy=True)
        u.rename("displacement", "")
        xdmf.write(u, t)
except Exception as e:
    logger.info(f"Runner error: {e}")
finally:
    if pv_file is not None:
        pv_file.close()
    xdmf.close()
    logger.info("Done.")

t_script_end = time.time()
logger.info(f"TOTAL SCRIPT WALLCLOCK TIME: {t_script_end - t_script_start:.2f}s "
            f"({(t_script_end - t_script_start)/60:.2f} min) "
            f"at {dolfin.MPI.size(dolfin.MPI.comm_world)} ranks")