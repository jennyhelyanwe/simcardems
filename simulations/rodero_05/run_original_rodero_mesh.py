"""
run_coarse.py — BiV cardiac electromechanics, coarse mesh. Orchestration
only — mechanics/checkpointing/runner logic now live in separate modules
(cavity_mechanics.py, checkpoint_coord.py, biv_runner.py, mesh_diagnostics.py,
pseudo_ecg.py), imported below.
"""

import os
import json
import time

cache_dir = os.environ.get("FENICS_CACHE_DIR", os.path.expanduser("~/.cache"))
os.environ["XDG_CACHE_HOME"] = cache_dir

import logging
logging.getLogger("simcardems.newton_solver").setLevel(logging.DEBUG)
logging.getLogger("simcardems.biv_cavity_cycle_controller").setLevel(logging.INFO)
logging.getLogger("simcardems.runner").setLevel(logging.DEBUG)
logging.getLogger("__main__").setLevel(logging.DEBUG)

import numpy as np
import pandas as pd

import dolfin
dolfin.PETScOptions.set("mat_mumps_icntl_4", "0")
dolfin.PETScOptions.set("snes_monitor")

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
from simcardems.time_stepper import TimeStepper
from simcardems.biv_cavity_cycle_controller import (
    BiVCycleController, CavityState, CycleParams, WindkesselParams,
)
from simcardems.geometry import refine_mesh, StimulusDomain
from simcardems.steady_state_cell_cache import load_steady_state_cache, apply_celltype_aware_initial_conditions
from simcardems import utils

from checkpoint_coord import load_function_by_coordinate, _field_stats, _leaf_stats_direct
from mesh_diagnostics import check_orthonormality, tet_volumes, tet_quality_ratios
from pseudo_ecg import PseudoECG
from cavity_mechanics import get_u_view, DualCavityModeManager
from biv_runner import BiVCycleRunner, apply_phase_dt

t_script_start = time.time()
logger = utils.getLogger(__name__)
logger.info(['dolfin:', dolfin.__version__])
import petsc4py; logger.info(['petsc4py:', petsc4py.__version__])
from petsc4py import PETSc; logger.info(['PETSc:', PETSc.Sys.getVersion()])

# ── Run-mode toggle ──────────────────────────────────────────────────────
RUN_MODE = os.environ.get("RUN_MODE", "local")
assert RUN_MODE in ("local", "archer2")
NUM_REFINEMENTS = 3 if RUN_MODE == "archer2" else 0
# NUM_REFINEMENTS = 1 # Force refinement to check valve mask bug
UNIFORM_ACTIVATION_TEST = (RUN_MODE == "local")
logger.info(f"RUN_MODE = {RUN_MODE} (NUM_REFINEMENTS={NUM_REFINEMENTS}, UNIFORM_ACTIVATION_TEST={UNIFORM_ACTIVATION_TEST})")

RESOLUTION = "fine"
MESH_DIR = "meshes/"
RUN_TAG = os.environ.get("RUN_TAG", "default")
RESULTS_DIR = f"results_{RESOLUTION}/{RUN_TAG}/"

# ── Warm start configuration ─────────────────────────────────────────────
WARM_START_T_MS = None #   # e.g. 100.0 to restart from t=100ms, or None for fresh start


# ── 1. Geometry ───────────────────────────────────────────────────────────
logger.info('Build geometry...')
geo = Geometry.from_file(MESH_DIR + "rodero_05_" + RESOLUTION + ".h5")

mesh = dolfin.Mesh()
with dolfin.HDF5File(mesh.mpi_comm(), MESH_DIR + "rodero_05_" + RESOLUTION + ".h5", "r") as f:
    f.read(mesh, "mesh", False)
coords = mesh.coordinates()
cells_arr = mesh.cells()
logger.info(f"Mesh: {mesh.num_vertices()} vertices, {mesh.num_cells()} cells")

# Mesh quality check
signed_vols = tet_volumes(coords, cells_arr, signed=True)
n_negative = np.sum(signed_vols < 0)
# n_zero = np.sum(np.abs(signed_vols) < 1e-10)
# logger.info(f"Reference mesh: {n_negative} elements with NEGATIVE signed volume (inverted winding), "
#             f"{n_zero} elements with ~ZERO volume (degenerate)")
# if n_negative > 0:
#     bad_idx = np.where(signed_vols < 0)[0]
#     bad_centers = coords[cells_arr[bad_idx]].mean(axis=1)
#     logger.info(f"First few inverted element centers:\n{bad_centers[:10]}")

if NUM_REFINEMENTS > 0:
    ep_mesh = refine_mesh(geo.mesh, num_refinements=NUM_REFINEMENTS)
    ffun_ep = dolfin.adapt(geo.ffun, ep_mesh)
    biv_geo = BiVentricularGeometry.from_geometry(geo, ep_mesh=ep_mesh, ffun_ep=ffun_ep,
                                                    parameters={"num_refinements": NUM_REFINEMENTS})
else:
    biv_geo = BiVentricularGeometry.from_geometry(geo, ep_mesh=geo.mesh, ffun_ep=geo.ffun)

logger.info(f"EP Mesh vertices: {biv_geo.ep_mesh.num_vertices()}")
logger.info(f"Mechanics Mesh vertices: {biv_geo.mechanics_mesh.num_vertices()}")

# Delineate different materials for assigning conduction velocity, stiffness, and contractility.
# coarse_tv = np.load(MESH_DIR + '/rodero_05_' + RESOLUTION + '_tv.npy')
tv = pd.read_csv(
    MESH_DIR + '/rodero_05_fine/rodero_05_fine_elementfield_tv-element.csv', header=None
).to_numpy().flatten().astype(int)
is_valve_float = (tv >= 7).astype(float) # Isolate valve plug elements
coarse_centres_all = np.array([cell.midpoint().array() for cell in dolfin.cells(biv_geo.mechanics_mesh)])
valve_fn = map_dense_field_to_ep_mesh(biv_geo.mechanics_mesh, coarse_centres_all, is_valve_float)
biv_geo.valve_mask = valve_fn

# Quality check on orthonormality of fibre vectors.
check_orthonormality(biv_geo.f0, biv_geo.s0, biv_geo.n0)

# ── 2. Activation times ───────────────────────────────────────────────────
node_coords = pd.read_csv(MESH_DIR + "/rodero_05_fine_xyz.csv", header=None).to_numpy() * 10.0 # Convert from cm to mm, because activation time fields are in Alya format (cm).
node_ids, activation_times, coords_subset = load_activation_times(MESH_DIR + "/heart.endocardial-activation-times", node_coords)
activation_times = activation_times * 1000.0 # convert from s to ms.

act_fn = interpolate_activation_to_ep_mesh(
    ep_mesh=biv_geo.ep_mesh,
    endo_marker_ep=[biv_geo.markers["ENDO_LV"][0], biv_geo.markers["ENDO_RV"][0]],
    ffun_ep=biv_geo.ffun_ep, node_ids_mech=node_ids,
    activation_times_mech=activation_times, coords_subset_mech=coords_subset,
)

if UNIFORM_ACTIVATION_TEST:
    act_fn.vector()[:] = 110
    whole_mesh_marker = dolfin.MeshFunction("size_t", biv_geo.ep_mesh, biv_geo.ep_mesh.topology().dim(), 1)
    biv_geo.stimulus_domain = StimulusDomain(domain=whole_mesh_marker, marker=1)
else:
    biv_geo.stimulus_domain = endocardial_stimulus_domain(
        mesh=biv_geo.ep_mesh, ffun=biv_geo.ffun_ep,
        endo_markers=[biv_geo.markers["ENDO_LV"][0], biv_geo.markers["ENDO_RV"][0]], layer_thickness=2,
    )

# # ── 2. Activation times — per-node local activation map from Eikonal solution (coarse mesh) ──
# logger.info('Load local activation time map (coarse mesh)...')
# activation_df = pd.read_csv(MESH_DIR + "rodero_05_coarse_" + RESOLUTION + "_lat.csv")
# node_ids_local = activation_df["node_id"].to_numpy()
# activation_times_local = activation_df["activation_time_ms"].to_numpy()
#
# order = np.argsort(node_ids_local)
# activation_times_sorted = activation_times_local[order]
# import h5py
# with h5py.File(MESH_DIR + "rodero_05_coarse_" + RESOLUTION + ".h5", "r") as f:
#     global_coords = f["mesh/coordinates"][:]  # adjust path once confirmed — GLOBAL, unpartitioned, identical on every rank
#
# coarse_coords_for_activation = global_coords[node_ids_local[order]]
#
# act_fn = map_dense_field_to_ep_mesh(biv_geo.ep_mesh, coarse_coords_for_activation, activation_times_sorted)
#
# whole_mesh_marker = dolfin.MeshFunction("size_t", biv_geo.ep_mesh, biv_geo.ep_mesh.topology().dim(), 1)
# biv_geo.stimulus_domain = StimulusDomain(domain=whole_mesh_marker, marker=1)

# ── Steady-state cell model cache ─────────────────────────────────────────
CELL_PARAMS_OVERRIDE = {}
PCL = 800.0
steady_state = load_steady_state_cache(CELL_PARAMS_OVERRIDE, PCL, max_beats=300)
cell_init_file = steady_state["cell_init_files"][0]

# ── 3. Config ──────────────────────────────────────────────────────────────
config = Config()
config.cell_init_file = cell_init_file
config.T = 800.0
config.dt = 10.0
config.dt_mech = 10.0
config.geometry_path = MESH_DIR + "rodero_05_" + RESOLUTION + ".h5"
config.outdir = RESULTS_DIR + "biv_coarse_run_output"
config.coupling_type = "fully_coupled_Tor_Land"
config.save_freq = 10
config.linear_mechanics_solver = "mumps"
config.spring = 500.0
config.traction = 0.005 # Initial pressure applied on cavities
config.mechanics_use_custom_newton_solver = True
config.mechanics_solve_strategy = "fixed"
config.mech_threshold = 1.0
config.relaxation_factor = 1.0 # dampening of Newton-Raphson iterations, to help with non-convergence and prevent overshooting.
# config.relaxation_factor = 0.2  # was 1.0 — full undamped step was overshooting
                                  # into inverted elements on the very first solve

# Material parameters.
# material_params_override = dict(a=2.28, a_f=1.686, b=9.726, b_f=15.779, a_s=0.0, b_s=0.0, a_fs=0.0, b_fs=0.0)
material_params_override = dict(a=0.61, a_f=1.56, b=7.5, b_f=35.31, a_s=0.7, b_s=33.24, a_fs=0.46, b_fs=5.09)
# material_params_override = dict(a=0.059, a_f=18.471, b=8.023, b_f=16.026, a_s=2.184, b_s=11.12, a_fs=0.216, b_fs=11.436)
# material_params_override = dict(
#     a=2.0, a_f=0.0, b=8.0, b_f=0.0,
#     a_s=0.0, b_s=0.0, a_fs=0.0, b_fs=0.0,
# )

# ── 4. Spatial fields ────────────────────────────────────────────────────
node_coords = pd.read_csv(MESH_DIR + "/rodero_05_fine_xyz.csv", header=None).to_numpy() * 10.0
ct_values = load_dense_node_field(MESH_DIR + "rodero_05_fine_nodefield_cell-type.csv")
cell_fn = map_dense_field_to_dg0_function(biv_geo.ep_mesh, node_coords, ct_values, {1: 0, 2: 2, 3: 1})
iks_values = load_dense_node_field(MESH_DIR + "rodero_05_fine_nodefield_sf_IKs.csv")
iks_fn = map_dense_field_to_ep_mesh(biv_geo.ep_mesh, node_coords, iks_values)

# ── 5. EM coupling ───────────────────────────────────────────────────────
VALVE_STIFFNESS_SCALE = 3.0 #  3.0
logger.info("" + "=" * 60)
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

coupling = em_model.setup_EM_model_from_config(
    config, geometry=biv_geo, activation_times=act_fn,
    celltype_function=cell_fn, iks_scale_function=iks_fn,
    material_parameters=material_params_override,
    valve_stiffness_scale=VALVE_STIFFNESS_SCALE,
)
mech_problem = coupling.mech_solver
mech_problem.solver.parameters["relative_tolerance"] = 1e-5   # was 1e-5
mech_problem.solver.parameters["absolute_tolerance"] = 1e-5
mech_problem.solver.parameters["linear_solver"] = "mumps"
mech_problem.solver.parameters["preconditioner"] = "lu"
logger.info(f"  [solver check] right after setting: linear_solver={mech_problem.solver.parameters['linear_solver']}, "
            f"preconditioner={mech_problem.solver.parameters['preconditioner']}")
# coupling.mech_solver.material.active._parameters["Tref"] = 0.0  # kill active tension entirely —
                                                                    # isolates whether Ta<->lambda
                                                                    # feedback is the source of chaos

# # ── Constrain rigid-body motion ────────────────────────────────────────
# # With only a Robin spring (normal-only) on EPI and a free BASE, nothing
# # resists tangential/rotational motion at all. A single apex pin removes
# # rigid TRANSLATION but not rigid ROTATION about that point — pin a
# # second point too, with only ONE component constrained there, to fix
# # rotation without over-constraining the mesh.
# mesh_ = biv_geo.mechanics_mesh
# mesh_.init(2, 0)
# ffun_arr_ = biv_geo.ffun.array()
# epi_marker_val = biv_geo.markers["EPI"][0]
# base_marker_val = biv_geo.markers["BASE"][0]
#
# epi_vertex_ids = set()
# base_vertex_ids = set()
# conn20 = mesh_.topology()(2, 0)
# for fidx in range(mesh_.num_entities(2)):
#     if ffun_arr_[fidx] == epi_marker_val:
#         epi_vertex_ids.update(conn20(fidx))
#     elif ffun_arr_[fidx] == base_marker_val:
#         base_vertex_ids.update(conn20(fidx))
#
# coords_ = mesh_.coordinates()
# base_centroid = coords_[list(base_vertex_ids)].mean(axis=0)
# epi_coords = coords_[list(epi_vertex_ids)]
# dists = np.linalg.norm(epi_coords - base_centroid, axis=1)
# apex_vertex_local = list(epi_vertex_ids)[int(np.argmax(dists))]
# apex_point = coords_[apex_vertex_local]
#
# # Second point: farthest EPI vertex from the apex, for the anti-rotation pin
# dists_from_apex = np.linalg.norm(epi_coords - apex_point, axis=1)
# second_vertex_local = list(epi_vertex_ids)[int(np.argmax(dists_from_apex))]
# second_point = coords_[second_vertex_local]
#
# logger.info(f"Apex pin at {apex_point} (full 3-component), "
#             f"anti-rotation pin at {second_point} (1 component)")
#
#
# def _apex_dirichlet_bc(W):
#     class ApexPoint(dolfin.SubDomain):
#         def inside(self, x, on_boundary):
#             return (dolfin.near(x[0], apex_point[0], 1e-6)
#                     and dolfin.near(x[1], apex_point[1], 1e-6)
#                     and dolfin.near(x[2], apex_point[2], 1e-6))
#
#     class SecondPoint(dolfin.SubDomain):
#         def inside(self, x, on_boundary):
#             return (dolfin.near(x[0], second_point[0], 1e-6)
#                     and dolfin.near(x[1], second_point[1], 1e-6)
#                     and dolfin.near(x[2], second_point[2], 1e-6))
#
#     V = W.sub(0)
#     return [
#         dolfin.DirichletBC(V, dolfin.Constant((0.0, 0.0, 0.0)), ApexPoint(), method="pointwise"),
#         dolfin.DirichletBC(V.sub(1), dolfin.Constant(0.0), SecondPoint(), method="pointwise"),
#     ]
#
#
# mech_problem.bcs.dirichlet = list(mech_problem.bcs.dirichlet) + [_apex_dirichlet_bc]
# mech_problem._set_dirichlet_bc()
# mech_problem._init_solver()
# logger.info("Apex + anti-rotation pins applied; solver rebuilt.")

state_names = list(TorLandFull.default_initial_conditions().keys())
apply_celltype_aware_initial_conditions(coupling, cell_fn, steady_state["cell_init_files"], state_names)

# ── 6. Cycle controller ───────────────────────────────────────────────────
LV_ENDO_MARKER = biv_geo.markers["ENDO_LV"][0]
RV_ENDO_MARKER = biv_geo.markers["ENDO_RV"][0]

lv_pressure_const, rv_pressure_const = None, None
for nbc in mech_problem.bcs.neumann:
    if nbc.marker == LV_ENDO_MARKER:
        lv_pressure_const = nbc.traction
    elif nbc.marker == RV_ENDO_MARKER:
        rv_pressure_const = nbc.traction
if lv_pressure_const is None or rv_pressure_const is None:
    raise RuntimeError("Missing Neumann BC on ENDO_LV/ENDO_RV")

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

u0 = get_u_view(mech_problem)
cycle_controller.initialize(u0)
cavity_manager = DualCavityModeManager(
    mech_problem, biv_geo, mech_problem.material,
    LV_ENDO_MARKER, RV_ENDO_MARKER, lv_pressure_const, rv_pressure_const,
)
logger.debug(f"Initial LV volume: {lv_state.volume_n:.2f}")
logger.debug(f"Initial RV volume: {rv_state.volume_n:.2f}")

# ── 7. Pseudo-ECG ─────────────────────────────────────────────────────────
electrode_df = pd.read_csv(MESH_DIR + "rodero_05_fine_nodefield_electrode_xyz.csv", header=None, names=["x", "y", "z"])
electrodes = electrode_df.to_numpy()
LA, RA, LL = electrodes[0], electrodes[1], electrodes[2]
V1, V2, V3, V4, V5, V6 = electrodes[4], electrodes[5], electrodes[6], electrodes[7], electrodes[8], electrodes[9]
wct = tuple((LA + RA + LL) / 3.0)

pseudo_ecg = PseudoECG(leads={
    "I": (tuple(LA), tuple(RA)), "II": (tuple(LL), tuple(RA)), "III": (tuple(LL), tuple(LA)),
    "aVR": (tuple(RA), tuple((LA + LL) / 2.0)), "aVL": (tuple(LA), tuple((RA + LL) / 2.0)),
    "aVF": (tuple(LL), tuple((RA + LA) / 2.0)),
    "V1": (tuple(V1), wct), "V2": (tuple(V2), wct), "V3": (tuple(V3), wct),
    "V4": (tuple(V4), wct), "V5": (tuple(V5), wct), "V6": (tuple(V6), wct),
}, sigma_b=1.0)

# ── 8. Run ────────────────────────────────────────────────────────────────
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
    state_names,
    pseudo_ecg=pseudo_ecg, dt_ecg=5.0,
    warm_start_freq=5.0, t_restart=t_restart,
)
runner._cavity_manager = cavity_manager

if WARM_START_T_MS is not None:
    checkpoint_base = WARM_START
    if os.path.exists(checkpoint_base + ".json"):
        logger.info(f"Loading warm start from t={WARM_START_T_MS:.1f} ms (coordinate-based)...")

        max_dist_vs = load_function_by_coordinate(coupling.ep_solver.vs, checkpoint_base + "_vs")
        load_function_by_coordinate(coupling.mech_solver.state, checkpoint_base + "_mechstate")
        load_function_by_coordinate(coupling.lmbda_mech, checkpoint_base + "_lmbda")
        load_function_by_coordinate(coupling.Zetas_mech, checkpoint_base + "_zetas")
        load_function_by_coordinate(coupling.Zetaw_mech, checkpoint_base + "_zetaw")
        load_function_by_coordinate(coupling.mech_solver.material.active.Ta_current, checkpoint_base + "_ta")
        coupling.ep_solver.vs_.assign(coupling.ep_solver.vs)
        coupling.mechanics_to_coupling()
        logger.info(f"  Coordinate-match max distance across all leaves (vs): {max_dist_vs:.3e}")

        # ── Sanity check: confirm restored cell-model state is physiological ──
        def _vs_field_max(name):
            key_map = {"cai": "Ca", "v": "V"}
            key = key_map.get(name, name)
            coupling.assigners.assign_ep()
            fn = coupling.assigners.functions["ep"][key]
            local_max = float(fn.vector().get_local().max()) if len(fn.vector().get_local()) > 0 else float('-inf')
            return dolfin.MPI.max(dolfin.MPI.comm_world, local_max)

        logger.info("  [warm-start sanity check] cell-model state (from vs):")
        garbage_detected = False
        for fname in ["cai", "XS", "XW", "CaTrpn", "TmB", "Cd"]:
            val = _vs_field_max(fname)
            logger.info(f"    {fname}_max = {val:.6e}")
            if abs(val) > 1e6:
                garbage_detected = True
                logger.info(f"    *** WARNING: {fname}_max looks like garbage (>1e6), "
                            f"restore may still be broken ***")

        active = coupling.mech_solver.material.active
        ta_current_max = dolfin.MPI.max(dolfin.MPI.comm_world, float(active.Ta_current.vector().max()))
        logger.info(f"    Ta_current_max (restored) = {ta_current_max:.6f}")

        if garbage_detected:
            raise RuntimeError("Warm-start restore produced garbage cell-model state — "
                                "do not proceed with this checkpoint.")
        logger.info("  [warm-start sanity check] PASSED — all fields physiological.")

        with open(checkpoint_base + ".json") as f:
            ws = json.load(f)

        # ── Checksum verification ──
        if "checksums" in ws:
            coupling.assigners.assign_ep()
            live_checksums = {
                "u": _field_stats(get_u_view(mech_problem)),
                "lmbda_mech": _field_stats(coupling.lmbda_mech),
                "Zetas_mech": _field_stats(coupling.Zetas_mech),
                "Zetaw_mech": _field_stats(coupling.Zetaw_mech),
                "Ta_current": _field_stats(coupling.mech_solver.material.active.Ta_current),
            }
            for fname in ["cai", "XS", "XW", "CaTrpn", "TmB", "Cd", "v"]:
                idx = state_names.index(fname)
                live_checksums[f"vs_{fname}"] = _leaf_stats_direct(coupling.ep_solver.vs, idx)

            logger.info("  [checkpoint verification] comparing restored state against save-time checksums:")
            all_ok = True
            for key, saved_stats in ws["checksums"].items():
                live_stats = live_checksums[key]
                for stat_name in ("max", "min", "mean"):
                    saved_val = saved_stats[stat_name]
                    live_val = live_stats[stat_name]
                    rel_diff = abs(live_val - saved_val) / max(abs(saved_val), 1e-12)
                    status = "OK" if rel_diff < 1e-6 else "MISMATCH"
                    if status == "MISMATCH":
                        all_ok = False
                    logger.info(f"    {key}.{stat_name}: saved={saved_val:.6e}, restored={live_val:.6e} [{status}]")

            if not all_ok:
                raise RuntimeError(
                    "Checkpoint verification FAILED — restored state does not match save-time checksums.")
            logger.info("  [checkpoint verification] PASSED — all fields match save-time checksums.")

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

    apply_phase_dt(config, cycle_controller.lv_state.phase, WARM_START_T_MS, time_stepper=None, logger=logger)
else:
    apply_phase_dt(config, cycle_controller.lv_state.phase, 0.0, time_stepper=None, logger=logger)


try:
    runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=False)
except Exception as e:
    logger.info(f"Runner error: {e}")
finally:
    runner.close_files()
    logger.info("Done.")

t_script_end = time.time()
logger.info(f"TOTAL SCRIPT WALLCLOCK TIME: {t_script_end - t_script_start:.2f}s "
            f"({(t_script_end - t_script_start)/60:.2f} min) at {dolfin.MPI.size(dolfin.MPI.comm_world)} ranks")
