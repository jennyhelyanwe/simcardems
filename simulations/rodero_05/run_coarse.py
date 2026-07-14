import os
cache_dir = os.environ.get("FENICS_CACHE_DIR", os.path.expanduser("~/.cache"))
os.environ["XDG_CACHE_HOME"] = cache_dir

import logging
logging.getLogger("simcardems.newton_solver").setLevel(logging.DEBUG)
logging.getLogger("simcardems.biv_cavity_cycle_controller").setLevel(logging.WARNING)
import dataclasses
import dolfin
dolfin.PETScOptions.set("mat_mumps_icntl_4", "0")
import numpy as np
import pandas as pd
import pulse
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

def mpi_print(*args, **kwargs):
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print(*args, **kwargs, flush=True)

RESOLUTION = "2mm"

# ── Warm start configuration ──────────────────────────────────────────────────
WARM_START_T_MS = None  # Set to e.g. 100.0 to restart from t=100ms, or None for fresh start

# ── PseudoECG ──────────────────────────────────────────────────────────────────

class PseudoECG:
    def __init__(self, leads: dict, sigma_b: float = 1.0):
        self.leads    = leads
        self.sigma_b  = sigma_b
        self.log      = {name: [] for name in leads}
        self.times    = []
        self._file    = None

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
        self._mech_problem     = mech_problem
        self._outdir           = outdir
        self._pseudo_ecg       = pseudo_ecg
        self._dt_ecg           = dt_ecg
        self._last_ecg_t       = -dt_ecg
        self._warm_start_freq = warm_start_freq
        self._last_warm_start_t = -warm_start_freq

        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
                pv_path = os.path.join(outdir, "pv_loop.csv")
                if os.path.exists(pv_path) and t_restart > 0:
                    # Truncate to t_restart
                    import pandas as pd
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
        mpi_print(f"  Mechanics solve time: {t1 - t0:.2f}s")
        self.coupling.update_prev_mechanics()
        self.coupling.mechanics_to_coupling()
        self.coupling.coupling_to_ep()

        t_ms  = TimeStepper.ns2ms(self.t)
        dt_ms = self._config.dt
        self._cycle_controller.step(
            problem=self._mech_problem,
            t=t_ms,
            dt=dt_ms,
        )
        lv = self._cycle_controller.lv_state
        rv = self._cycle_controller.rv_state

        mpi_print(
            f"  → t={t_ms:.1f} ms"
            f"  LV phase={lv.phase}  LVP={lv.pressure_n:.3f} kPa  LVV={lv.volume_n/1000:.2f} mL"
            f"  RV phase={rv.phase}  RVP={rv.pressure_n:.3f} kPa  RVV={rv.volume_n/1000:.2f} mL"
        )
        # Save warm start
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
                import json, dataclasses
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
        if hasattr(x, "vector"):  # dolfin Function
            return x.vector().get_local().reshape(-1, 3)
        return x  # already a numpy array

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
mpi_print('Build geometry...')
from cardiac_geometries.geometry import Geometry
from simcardems.bivgeometry import BiVentricularGeometry
from simcardems.geometry import refine_mesh

geo = Geometry.from_file("rodero_05_coarse_"+RESOLUTION+".h5")

import dolfin
import numpy as np

mesh = dolfin.Mesh()
with dolfin.HDF5File(mesh.mpi_comm(), "rodero_05_coarse_2mm.h5", "r") as f:
    f.read(mesh, "mesh", False)

coords = mesh.coordinates()
cells_arr = mesh.cells()

mpi_print(f"Mesh: {mesh.num_vertices()} vertices, {mesh.num_cells()} cells")
mpi_print(f"cells() array shape: {cells_arr.shape}")

# ── Volumes ──────────────────────────────────────────────────────────────
def tet_volumes(coords, cells_arr):
    p0 = coords[cells_arr[:, 0]]
    p1 = coords[cells_arr[:, 1]]
    p2 = coords[cells_arr[:, 2]]
    p3 = coords[cells_arr[:, 3]]
    return np.abs(np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0

volumes = tet_volumes(coords, cells_arr)
mpi_print(f"Cell volumes: min={volumes.min():.4f}, mean={volumes.mean():.4f}, max={volumes.max():.4f}")
mpi_print(f"Negative volumes: {np.sum(volumes < 0)}")
mpi_print(f"Very small volumes (<0.01): {np.sum(volumes < 0.01)}")

# ── Inradius/circumradius quality ratio ──────────────────────────────────
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
        (a*a1 + b*b1 + c*c1) *
        (a*a1 + b*b1 - c*c1) *
        (a*a1 - b*b1 + c*c1) *
        (-a*a1 + b*b1 + c*c1)
    )
    circumradius = num / (24 * vol)

    return inradius / circumradius

radii = tet_quality_ratios(coords, cells_arr)
mpi_print(f"Inradius/circumradius: min={radii.min():.4f}, mean={radii.mean():.4f}")
mpi_print(f"Poor quality (ratio < 0.1): {np.sum(radii < 0.1)}")
mpi_print(f"Poor quality (ratio < 0.05): {np.sum(radii < 0.05)}")
mpi_print(f"Poor quality (ratio < 0.02): {np.sum(radii < 0.02)}")

# Worst offenders, in case a handful of degenerate elements dominate
worst_idx = np.argsort(radii)[:10]
mpi_print(f"\nWorst 10 elements (lowest quality ratio):")
for i in worst_idx:
    mpi_print(f"  cell {i}: ratio={radii[i]:.5f}, volume={volumes[i]:.5f}")

worst_centres = coords[cells_arr[worst_idx]].mean(axis=1)

mpi_print("Worst 14 element centroids:")
for i, c in zip(worst_idx, worst_centres):
    mpi_print(f"  cell {i}: centroid={c}, ratio={radii[i]:.5f}")

# Compare against mesh bounding box / base plane to see if these cluster near the base
mpi_print(f"\nMesh coordinate ranges:")
mpi_print(f"  x: {coords[:,0].min():.2f} to {coords[:,0].max():.2f}")
mpi_print(f"  y: {coords[:,1].min():.2f} to {coords[:,1].max():.2f}")
mpi_print(f"  z: {coords[:,2].min():.2f} to {coords[:,2].max():.2f}")

# Build refined EP mesh with parent tracking
ep_mesh = refine_mesh(geo.mesh, num_refinements=1)
ffun_ep = dolfin.adapt(geo.ffun, ep_mesh)

#biv_geo = BiVentricularGeometry.from_geometry(
#    geo,
#    ep_mesh=ep_mesh,
#    ffun_ep=ffun_ep,
#    parameters={"num_refinements": 1},
#)

biv_geo = BiVentricularGeometry.from_geometry(geo, ep_mesh=geo.mesh, ffun_ep=geo.ffun)

mpi_print(f"EP Mesh vertices: {biv_geo.ep_mesh.num_vertices()}")
mpi_print(f"Mechanics Mesh vertices: {biv_geo.mechanics_mesh.num_vertices()}")

import numpy as np
from scipy.spatial import cKDTree

# Load valve plug mask
coarse_tv = np.load('./rodero_05_coarse_'+RESOLUTION+'_tv.npy')
is_valve = (coarse_tv >= 7).astype(float)  # 1.0 for valve, 0.0 for myocardium

# Create DG0 function for valve mask
from simcardems.spatial_fields import map_dense_field_to_dg0_function

# Use coarse cell centres as the "dense field" coordinates
coarse_centres_all = np.array([cell.midpoint().array()
                                for cell in dolfin.cells(biv_geo.mechanics_mesh)])
is_valve_float = (coarse_tv >= 7).astype(float)

valve_fn = map_dense_field_to_ep_mesh(
    biv_geo.mechanics_mesh,
    coarse_centres_all,
    is_valve_float,
)

mpi_print(f'Mechanics valve plug elements: {int(is_valve.sum())}')
biv_geo.valve_mask = valve_fn

f0 = biv_geo.f0
s0 = biv_geo.s0
n0 = biv_geo.n0

results = check_orthonormality(f0, s0, n0)

epi_marker = geo.markers["EPI"][0]
base_marker = geo.markers["BASE"][0]
ffun_arr = geo.ffun.array()
mpi_print(f"EPI marker {epi_marker}: {np.sum(ffun_arr == epi_marker)} facets")
mpi_print(f"BASE marker {base_marker}: {np.sum(ffun_arr == base_marker)} facets")
mpi_print(f"Total facets: {len(ffun_arr)}")

# ── 2. Activation times ────────────────────────────────────────────────────────

mpi_print('Load activation times...')
node_coords = pd.read_csv(
    "./rodero_05_fine/rodero_05_fine_xyz.csv", header=None
).to_numpy() * 10.0

node_ids, activation_times, coords_subset = load_activation_times(
    "./heart.endocardial-activation-times", node_coords
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

mpi_print('Create stimulus domain...')
stim_domain = endocardial_stimulus_domain(
    mesh=biv_geo.ep_mesh,
    ffun=biv_geo.ffun_ep,
    endo_markers=[biv_geo.markers["ENDO_LV"][0], biv_geo.markers["ENDO_RV"][0]],
    layer_thickness=2,
)
biv_geo.stimulus_domain = stim_domain


# ── 4. Config ──────────────────────────────────────────────────────────────────

mpi_print('Configuring...')
config = Config()
config.T                                  = 800.0
config.dt                                 = 1.0
config.dt_mech                            = 5.0
config.geometry_path                      = "rodero_05_coarse_"+RESOLUTION+".h5"
config.outdir                             = "biv_coarse_run_output"
config.coupling_type                      = "fully_coupled_Tor_Land"
config.save_freq                          = 20
config.linear_mechanics_solver            = "mumps"
config.spring                             = 1000.0
config.traction                           = 0.001
config.mechanics_use_custom_newton_solver = True
config.mechanics_solve_strategy           = "hybrid"
config.mech_threshold                     = 1.0
config.relaxation_factor                  = 1.0


# ── 5. Spatial fields ──────────────────────────────────────────────────────────

mpi_print('Load cell type...')
ct_values = load_dense_node_field("./rodero_05_fine/rodero_05_fine_nodefield_cell-type.csv")
cell_fn = map_dense_field_to_dg0_function(
    biv_geo.ep_mesh, node_coords, ct_values, {1: 0, 2: 2, 3: 1}
)

mpi_print('Load sf IKs...')
iks_values = load_dense_node_field("./rodero_05_fine/rodero_05_fine_nodefield_sf_IKs.csv")
iks_fn = map_dense_field_to_ep_mesh(biv_geo.ep_mesh, node_coords, iks_values)


# ── 6. EM coupling ─────────────────────────────────────────────────────────────

mpi_print('Setting up EM model...')
coupling = em_model.setup_EM_model_from_config(
    config,
    geometry=biv_geo,
    activation_times=act_fn,
    celltype_function=cell_fn,
    iks_scale_function=iks_fn,
)

mpi_print(f"----------------------------State space dim: {coupling.mech_solver.state_space.dim()}")

mech_problem = coupling.mech_solver
mpi_print(f"BCs at preload: neumann={len(mech_problem.bcs.neumann)} robin={len(mech_problem.bcs.robin)} dirichlet={len(mech_problem.bcs.dirichlet)}")
for nbc in mech_problem.bcs.neumann:
    mpi_print(f"  Neumann: marker={nbc.marker} traction={float(nbc.traction):.6e}")
for rbc in mech_problem.bcs.robin:
    mpi_print(f"  Robin: marker={rbc.marker} value={float(rbc.value):.6e}")
for dbc in mech_problem.bcs.dirichlet:
    mpi_print(f"  Dirichlet: {dbc}")
mpi_print(f"  _dirichlet_bc: {mech_problem._dirichlet_bc}")


# ── 7. Cycle controller ────────────────────────────────────────────────────────

LV_ENDO_MARKER = biv_geo.markers["ENDO_LV"][0]
RV_ENDO_MARKER = biv_geo.markers["ENDO_RV"][0]
mech_problem   = coupling.mech_solver

lv_pressure_const = None
rv_pressure_const = None

for nbc in mech_problem.bcs.neumann:
    mpi_print(f"Found Neumann BC: marker={nbc.marker} traction={float(nbc.traction):.6e}")
    if nbc.marker == LV_ENDO_MARKER:
        lv_pressure_const = nbc.traction
    elif nbc.marker == RV_ENDO_MARKER:
        rv_pressure_const = nbc.traction

mpi_print(f"Total Neumann BCs: {len(mech_problem.bcs.neumann)}")

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
    gain_contraction=(0.01, 0.0),  # was (1.0, 0.5)
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
mpi_print(f"Initial LV volume: {lv_state.volume_n:.2f}")
mpi_print(f"Initial RV volume: {rv_state.volume_n:.2f}")


# ── 8. Pseudo-ECG ─────────────────────────────────────────────────────────────

mpi_print('Loading electrode locations...')
electrode_df = pd.read_csv(
    "./rodero_05_fine/rodero_05_fine_nodefield_electrode_xyz.csv",
    header=None, names=["x", "y", "z"],
)
electrodes = electrode_df.to_numpy()

LA, RA, LL = electrodes[0], electrodes[1], electrodes[2]
V1, V2, V3, V4, V5, V6 = electrodes[4], electrodes[5], electrodes[6], \
                           electrodes[7], electrodes[8], electrodes[9]
wct = tuple((LA + RA + LL) / 3.0)

pseudo_ecg = PseudoECG(
    leads={
        "I":   (tuple(LA), tuple(RA)),
        "II":  (tuple(LL), tuple(RA)),
        "III": (tuple(LL), tuple(LA)),
        "aVR": (tuple(RA), tuple((LA + LL) / 2.0)),
        "aVL": (tuple(LA), tuple((RA + LL) / 2.0)),
        "aVF": (tuple(LL), tuple((RA + LA) / 2.0)),
        "V1":  (tuple(V1), wct),
        "V2":  (tuple(V2), wct),
        "V3":  (tuple(V3), wct),
        "V4":  (tuple(V4), wct),
        "V5":  (tuple(V5), wct),
        "V6":  (tuple(V6), wct),
    },
    sigma_b=1.0,
)


# ── 9. Run ─────────────────────────────────────────────────────────────────────

os.makedirs(config.outdir, exist_ok=True)
config.traction = 0.0 #  float(lv_pressure_const)

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

bc = mech_problem._dirichlet_bc
if isinstance(bc, list):
    mpi_print(f"_dirichlet_bc is a list of {len(bc)} BCs")
    total = 0
    for i, b in enumerate(bc):
        n = len(b.get_boundary_values())
        mpi_print(f"  BC {i}: {n} dofs")
        total += n
    mpi_print(f"  Total constrained dofs: {total}")
else:
    mpi_print(f"_dirichlet_bc is a single BC with {len(bc.get_boundary_values())} dofs")
mpi_print(f"Expected EPI+BASE dofs: {len(dolfin.DirichletBC(mech_problem.state_space.sub(0), dolfin.Constant((0,0,0)), biv_geo.ffun, biv_geo.markers['EPI'][0]).get_boundary_values()) + len(dolfin.DirichletBC(mech_problem.state_space.sub(0), dolfin.Constant((0,0,0)), biv_geo.ffun, biv_geo.markers['BASE'][0]).get_boundary_values())}")

# Warm start read in fields.
if WARM_START_T_MS is not None:
    warm_start_name = f"warm_start_{int(WARM_START_T_MS):04d}ms"
    WARM_START = os.path.join(config.outdir, warm_start_name)
    if os.path.exists(WARM_START + ".h5") and os.path.exists(WARM_START + ".json"):
        mpi_print(f"Loading warm start from t={WARM_START_T_MS:.1f} ms...")
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
        mpi_print(f"Restarting from t={t_restart:.1f} ms")
    else:
        mpi_print(f"WARNING: warm start file not found for t={WARM_START_T_MS:.1f} ms")

# Run simulation.
try:
    runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=True)
except Exception as e:
    mpi_print(f"Runner error: {e}")
finally:
    runner.close_files()
    mpi_print("Done.")
