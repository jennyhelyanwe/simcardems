import os

cache_dir = os.environ.get("FENICS_CACHE_DIR", os.path.expanduser("~/.cache"))
os.environ["XDG_CACHE_HOME"] = cache_dir

import logging
logging.getLogger("simcardems.newton_solver").setLevel(logging.DEBUG)
logging.getLogger("simcardems.biv_cavity_cycle_controller").setLevel(logging.WARNING)
logging.getLogger("simcardems.runner").setLevel(logging.DEBUG)
logging.getLogger("simcardems.models.fully_coupled_Tor_Land.em_model").setLevel(logging.DEBUG)
import dataclasses
import dolfin
dolfin.PETScOptions.set("mat_mumps_icntl_4", "0")
import numpy as np
import pandas as pd
import pulse
from scipy.spatial import cKDTree
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
from simcardems.runner import Runner
from simcardems.time_stepper import TimeStepper
from simcardems.biv_cavity_cycle_controller import (
    BiVCycleController,
    CavityState,
    CycleParams,
    WindkesselParams,
    Phase,
)
from simcardems.postprocess import ecg_recovery
from simcardems.geometry import refine_mesh
from simcardems import utils
logger = utils.getLogger(__name__)

def mpi_print(*args, **kwargs):
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print(*args, **kwargs, flush=True)


logger.info(['dolfin:', dolfin.__version__])
import petsc4py; logger.info(['petsc4py:', petsc4py.__version__])
from petsc4py import PETSc; logger.info(['PETSc:', PETSc.Sys.getVersion()])

RESOLUTION = "4mm"
MESH_DIR = "meshes/"
RESULTS_DIR = "results/"

# ── Warm start configuration ──────────────────────────────────────────────────
WARM_START_T_MS = None  # Set to e.g. 100.0 to restart from t=100ms, or None for fresh start

# ── PseudoECG ──────────────────────────────────────────────────────────────────

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


# ── BiVCycleRunner ─────────────────────────────────────────────────────────────

class BiVCycleRunner(Runner):
    def set_cycle_controller(self, controller, mech_problem, outdir,
                              pseudo_ecg=None, dt_ecg=5.0,
                              warm_start_freq=5.0, t_restart=0.0):
        self._cycle_controller = controller
        self._mech_problem = mech_problem
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
        import time
        t0 = time.time()
        self.coupling.solve_mechanics()
        t1 = time.time()
        logger.debug(f"  Mechanics solve time: {t1 - t0:.2f}s")
        self.coupling.update_prev_mechanics()
        self.coupling.mechanics_to_coupling()
        self.coupling.coupling_to_ep()

        t_ms = TimeStepper.ns2ms(self.t)
        dt_ms = self._config.dt
        self._cycle_controller.step(
            problem=self._mech_problem,
            t=t_ms,
            dt=dt_ms,
        )
        lv = self._cycle_controller.lv_state
        rv = self._cycle_controller.rv_state

        logger.debug(
            f"  → t={t_ms:.1f} ms"
            f"  LV phase={lv.phase}  LVP={lv.pressure_n:.3f} kPa  LVV={lv.volume_n/1000:.2f} mL"
            f"  RV phase={rv.phase}  RVP={rv.pressure_n:.3f} kPa  RVV={rv.volume_n/1000:.2f} mL"
        )
        t_ms = TimeStepper.ns2ms(self.t)
        if (t_ms - self._last_warm_start_t) >= (self._warm_start_freq - 1e-10):
            checkpoint_name = f"warm_start_{int(t_ms):04d}ms"
            with dolfin.HDF5File(dolfin.MPI.comm_world, os.path.join(self._outdir, f"{checkpoint_name}.h5"), "w") as f:
                f.write(self.coupling.ep_solver.vs, "/ep/vs")
                f.write(self.coupling.mech_solver.state, "/mechanics/state")
                f.write(self.coupling.lmbda_mech, "/em/lmbda_prev")
                f.write(self.coupling.Zetas_mech, "/em/Zetas_prev")
                f.write(self.coupling.Zetaw_mech, "/em/Zetaw_prev")

            if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
                import json
                with open(os.path.join(self._outdir, f"{checkpoint_name}.json"), "w") as f:
                    json.dump({
                        "t_ms": TimeStepper.ns2ms(self.t),
                        "lv": {k: v for k, v in dataclasses.asdict(self._cycle_controller.lv_state).items() if
                               not isinstance(v, dict)},
                        "rv": {k: v for k, v in dataclasses.asdict(self._cycle_controller.rv_state).items() if
                               not isinstance(v, dict)},
                    }, f, indent=2)
            self._last_warm_start_t = t_ms

        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0 and self._pv_file is not None:
            self._pv_file.write(
                f"{t_ms:.3f},{lv.pressure_n:.6f},{lv.volume_n:.6f},"
                f"{rv.pressure_n:.6f},{rv.volume_n:.6f}\n"
            )
            self._pv_file.flush()

        if self._pseudo_ecg is not None and t_ms - self._last_ecg_t >= self._dt_ecg - 1e-10:
            self.coupling.assigners.assign_ep()
            v_fn = self.coupling.assigners.functions["ep"]["V"]
            self._pseudo_ecg.compute(v_fn, t_ms)
            self._last_ecg_t = t_ms

    def close_files(self):
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0 and self._pv_file is not None:
            self._pv_file.close()
        if self._pseudo_ecg is not None:
            self._pseudo_ecg.close()


def check_orthonormality(f0, s0, n0, label=""):
    def as_array(x):
        if hasattr(x, "vector"):
            return x.vector().get_local().reshape(-1, 3)
        return x

    F = as_array(f0)
    S = as_array(s0)
    N = as_array(n0)

    f0n = np.linalg.norm(F, axis=1)
    s0n = np.linalg.norm(S, axis=1)
    n0n = np.linalg.norm(N, axis=1)
    dot_fs = np.abs(np.sum(F * S, axis=1))
    dot_fn = np.abs(np.sum(F * N, axis=1))
    dot_sn = np.abs(np.sum(S * N, axis=1))
    print(f"{label} f0 norm: {f0n.min():.6f}-{f0n.max():.6f}, "
          f"max|f0.s0|={dot_fs.max():.3e}, max|f0.n0|={dot_fn.max():.3e}, max|s0.n0|={dot_sn.max():.3e}")


# ── 1. Geometry ────────────────────────────────────────────────────────────────
logger.info('Build geometry...')
from cardiac_geometries.geometry import Geometry
from simcardems.bivgeometry import BiVentricularGeometry

geo = Geometry.from_file(MESH_DIR + "rodero_05_coarse_" + RESOLUTION + ".h5")

mesh = dolfin.Mesh()
with dolfin.HDF5File(mesh.mpi_comm(), MESH_DIR + "rodero_05_coarse_" + RESOLUTION + ".h5", "r") as f:
    f.read(mesh, "mesh", False)

coords = mesh.coordinates()
cells_arr = mesh.cells()

logger.info(f"Mesh: {mesh.num_vertices()} vertices, {mesh.num_cells()} cells")

def tet_volumes(coords, cells_arr):
    p0 = coords[cells_arr[:, 0]]
    p1 = coords[cells_arr[:, 1]]
    p2 = coords[cells_arr[:, 2]]
    p3 = coords[cells_arr[:, 3]]
    return np.abs(np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0


volumes = tet_volumes(coords, cells_arr)
logger.debug(f"Cell volumes: min={volumes.min():.4f}, mean={volumes.mean():.4f}, max={volumes.max():.4f}")
logger.debug(f"Negative volumes: {np.sum(volumes < 0)}")
logger.debug(f"Very small volumes (<0.01): {np.sum(volumes < 0.01)}")


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


radii = tet_quality_ratios(coords, cells_arr)
logger.debug(f"Inradius/circumradius: min={radii.min():.4f}, mean={radii.mean():.4f}")
logger.debug(f"Poor quality (ratio < 0.1): {np.sum(radii < 0.1)}")
logger.debug(f"Poor quality (ratio < 0.05): {np.sum(radii < 0.05)}")
logger.debug(f"Poor quality (ratio < 0.02): {np.sum(radii < 0.02)}")


# Build refined EP mesh with parent tracking
NUM_REFINEMENTS = 0
if NUM_REFINEMENTS > 1:
    ep_mesh = refine_mesh(geo.mesh, num_refinements=NUM_REFINEMENTS)
    ffun_ep = dolfin.adapt(geo.ffun, ep_mesh)

    biv_geo = BiVentricularGeometry.from_geometry(
        geo,
        ep_mesh=ep_mesh,
        ffun_ep=ffun_ep,
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

valve_fn = map_dense_field_to_ep_mesh(
    biv_geo.mechanics_mesh,
    coarse_centres_all,
    is_valve_float,
)

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

# ── 2. Activation times ────────────────────────────────────────────────────────

logger.info('Load activation times...')
node_coords = pd.read_csv(
    MESH_DIR + "/rodero_05_fine_xyz.csv", header=None
).to_numpy() * 10.0

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

# ── 3. Stimulus domain ─────────────────────────────────────────────────────────

logger.info('Create stimulus domain...')
stim_domain = endocardial_stimulus_domain(
    mesh=biv_geo.ep_mesh,
    ffun=biv_geo.ffun_ep,
    endo_markers=[biv_geo.markers["ENDO_LV"][0], biv_geo.markers["ENDO_RV"][0]],
    layer_thickness=2,
)
biv_geo.stimulus_domain = stim_domain

# ── 4. Config ──────────────────────────────────────────────────────────────────

logger.info('Configuring...')
config = Config()
config.T = 800.0
config.dt = 0.5
config.dt_mech = 2.5
config.geometry_path = MESH_DIR + "rodero_05_coarse_" + RESOLUTION + ".h5"
config.outdir = RESULTS_DIR + "biv_coarse_run_output"
config.coupling_type = "fully_coupled_Tor_Land"
config.save_freq = 20
config.linear_mechanics_solver = "mumps"
config.spring = 100.0
config.traction = 0.001
config.mechanics_use_custom_newton_solver = True
config.mechanics_solve_strategy = "hybrid"
config.mech_threshold = 1.0
config.relaxation_factor = 0.3
# Scalability test
SCALABILITY_TEST = os.environ.get("SCALABILITY_TEST", "0") == "1"
if SCALABILITY_TEST:
    config.T = 20.0  # just enough for a handful of EP+mechanics steps
    config.save_freq = 1000.0  # effectively disable checkpoint I/O, which would skew timing

# Reverted (transversely isotropic) material parameters - known-good baseline.
# The full orthotropic set is a separate, still-open experiment - see notes.
material_params_override = dict(
    a=2.28,
    a_f=1.686,
    b=9.726,
    b_f=15.779,
    a_s=0.0,
    b_s=0.0,
    a_fs=0.0,
    b_fs=0.0,
)

# MATERIAL_SCALE = 1
# material_params_override = dict(
#     a=0.61 * MATERIAL_SCALE,
#     a_f=1.56* MATERIAL_SCALE,
#     b=7.5,
#     b_f=35.31,
#     a_s=0.70* MATERIAL_SCALE,
#     b_s=33.24,
#     a_fs=0.46* MATERIAL_SCALE,
#     b_fs=5.09,
# )
# material_params_override = dict(
#     a=0.059,
#     b=0.023,
#     a_f=18.472,
#     b_f=16.026,
#     a_s=2.481,
#     b_s=11.120,
#     a_fs=0.216,
#     b_fs=11.436,
# )

# ── 5. Spatial fields ──────────────────────────────────────────────────────────

logger.info('Load cell type...')
ct_values = load_dense_node_field(MESH_DIR + "rodero_05_fine_nodefield_cell-type.csv")
cell_fn = map_dense_field_to_dg0_function(
    biv_geo.ep_mesh, node_coords, ct_values, {1: 0, 2: 2, 3: 1}
)

logger.info('Load sf IKs...')
iks_values = load_dense_node_field(MESH_DIR + "rodero_05_fine_nodefield_sf_IKs.csv")
iks_fn = map_dense_field_to_ep_mesh(biv_geo.ep_mesh, node_coords, iks_values)

# ── 6. EM coupling ─────────────────────────────────────────────────────────────
# ── Parameter summary: print everything that affects the solve, up front ──
VALVE_STIFFNESS_SCALE = 5.0
logger.info("" + "=" * 60)
logger.info("RUN PARAMETERS")
logger.info("=" * 60)
logger.info(f"Resolution:                {RESOLUTION}")
logger.info(f"T (total time):            {config.T} ms")
logger.info(f"dt (EP):                   {config.dt} ms")
logger.info(f"dt_mech:                   {config.dt_mech} ms")
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

# ── 7. Cycle controller ────────────────────────────────────────────────────────

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
    t_zero=50.0,
    t_prestress=0.0,
    preload_pressure=0.5,
    prestress_pressure=0.0,
    t_end_diastole=130.0,
    p_end_diastole=1.3,
    gain_contraction=(0.01, 0.0),
    gain_relaxation=(0.05, 0.01),
    p_fill=0.1,
    period=800.0,
    windkessel=WindkesselParams(
        p_init=9.0,
        compliance=1.0,
        resistance=100.0,
        evolve=True,
    ),
)

rv_params = CycleParams(
    t_zero=50.0,
    t_prestress=0.0,
    preload_pressure=0.17,
    prestress_pressure=0.0,
    t_end_diastole=100.0,
    p_end_diastole=0.2,
    gain_contraction=(0.01, 0.0),
    gain_relaxation=(0.5, 0.2),
    p_fill=0.033,
    period=800.0,
    windkessel=WindkesselParams(
        p_init=3.0,
        compliance=3.0,
        resistance=33.0,
        evolve=True,
    ),
)

lv_state = CavityState(name="LV", params=lv_params)
rv_state = CavityState(name="RV", params=rv_params)

cycle_controller = BiVCycleController(
    lv_state=lv_state,
    rv_state=rv_state,
    lv_pressure_constant=lv_pressure_const,
    rv_pressure_constant=rv_pressure_const,
    geometry=biv_geo,
    lv_marker=LV_ENDO_MARKER,
    rv_marker=RV_ENDO_MARKER,
)

u0, _ = mech_problem.state.split(deepcopy=True)
cycle_controller.initialize(u0)
logger.debug(f"Initial LV volume: {lv_state.volume_n:.2f}")
logger.debug(f"Initial RV volume: {rv_state.volume_n:.2f}")

# ── 8. Pseudo-ECG ─────────────────────────────────────────────────────────────

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
        "V1": (tuple(V1), wct),
        "V2": (tuple(V2), wct),
        "V3": (tuple(V3), wct),
        "V4": (tuple(V4), wct),
        "V5": (tuple(V5), wct),
        "V6": (tuple(V6), wct),
    },
    sigma_b=1.0,
)

# ── 9. Run ─────────────────────────────────────────────────────────────────────

os.makedirs(config.outdir, exist_ok=True)
config.traction = 0.0

runner = BiVCycleRunner.from_models(coupling=coupling, config=config)

import json
t_restart = 0.0
if WARM_START_T_MS is not None:
    warm_start_name = f"warm_start_{int(WARM_START_T_MS):04d}ms"
    WARM_START = os.path.join(config.outdir, warm_start_name)
    with open(WARM_START + ".json") as f:
        ws = json.load(f)
    t_restart = ws["t_ms"]

runner.set_cycle_controller(
    cycle_controller, mech_problem, config.outdir,
    pseudo_ecg=pseudo_ecg, dt_ecg=5.0,
    warm_start_freq=5.0,
    t_restart=t_restart,
)

# bc = mech_problem._dirichlet_bc
# if isinstance(bc, list):
#     mpi_print(f"_dirichlet_bc is a list of {len(bc)} BCs")
#     total = 0
#     for i, b in enumerate(bc):
#         n = len(b.get_boundary_values())
#         mpi_print(f"  BC {i}: {n} dofs")
#         total += n
#     mpi_print(f"  Total constrained dofs: {total}")
# else:
#     mpi_print(f"_dirichlet_bc is a single BC with {len(bc.get_boundary_values())} dofs")
# mpi_print(f"Expected EPI+BASE dofs: {len(dolfin.DirichletBC(mech_problem.state_space.sub(0), dolfin.Constant((0, 0, 0)), biv_geo.ffun, biv_geo.markers['EPI'][0]).get_boundary_values()) + len(dolfin.DirichletBC(mech_problem.state_space.sub(0), dolfin.Constant((0, 0, 0)), biv_geo.ffun, biv_geo.markers['BASE'][0]).get_boundary_values())}")

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

try:
    runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=False)
except Exception as e:
    logger.info(f"Runner error: {e}")
finally:
    runner.close_files()
    logger.info("Done.")