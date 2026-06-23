"""
run_lv_ellipsoid.py
-------------------
LV ellipsoid cardiac cycle test — simcardems biv-dev branch.

Goals:
  - Load an LV ellipsoid geometry via cardiac_geometries
  - Simultaneous whole-mesh stimulus at end-diastole via uniform activation map
  - LV-only 5-phase cycle controller (BiVCycleController with stub RV)
  - MUMPS linear solver
  - Epicardial Robin spring BC, free base (no Dirichlet)
  - Fast end-to-end proof of concept (~800 ms = one beat)

Run locally:
  docker run --rm -it -v $(pwd):/home/shared -w /home/shared \
    ghcr.io/computationalphysiology/simcardems:latest \
    python run_lv_ellipsoid.py
"""

import os
cache_dir = os.environ.get("FENICS_CACHE_DIR", os.path.expanduser("~/.cache"))
os.environ["XDG_CACHE_HOME"] = cache_dir

import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)


import dolfin
dolfin.set_log_level(dolfin.LogLevel.DEBUG)
# dolfin.PETScOptions.set("ksp_monitor_true_residual")
dolfin.PETScOptions.set("ksp_type", "gmres")
dolfin.PETScOptions.set("pc_type", "jacobi")
# dolfin.PETScOptions.set("ksp_monitor_true_residual", "")
# dolfin.PETScOptions.set("ksp_type", "preonly")
# dolfin.PETScOptions.set("pc_type", "lu")
# dolfin.PETScOptions.set("pc_factor_mat_solver_type", "mumps")
# Force superlu_dist or lu for single-process debugging
# dolfin.PETScOptions.set("pc_factor_mat_solver_type", "superlu_dist")

import pulse
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
    compute_cavity_volume,
)

def mpi_print(*args, **kwargs):
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print(*args, **kwargs, flush=True)

# ── LVCycleRunner ──────────────────────────────────────────────────────────────

class LVCycleRunner(Runner):
    def set_cycle_controller(self, controller, mech_problem, outdir):
        self._cycle_controller = controller
        self._mech_problem = mech_problem
        self._outdir = outdir
        self._pv_log = []

    def _solve_mechanics(self):
        self.coupling.coupling_to_mechanics()
        self.coupling.solve_mechanics()
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
        state = self._cycle_controller.lv_state
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            from tqdm import tqdm
            tqdm.write(
                f"  → t={t_ms:.1f} ms  phase={state.phase}"
                f"  LVP={state.pressure_n:.3f} kPa"
                f"  LVV={state.volume_n:.2f}"
            )
        self._pv_log.append((t_ms, state.pressure_n, state.volume_n))

    def save_pv_log(self):
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            pv_arr = np.array(self._pv_log)
            path = os.path.join(self._outdir, "pv_loop.csv")
            np.savetxt(path, pv_arr, delimiter=",",
                       header="t_ms,LVP_kPa,LVV", comments="")
            mpi_print(f"PV loop saved to {path}")

# ── 1. Generate / load LV ellipsoid geometry ───────────────────────────────────

# GEO_PATH   = "our_geo.h5"
# GEO_SCHEMA = "our_geo.json"
GEO_PATH   = "/repo/simcardems-dev/demos/geometries/lv_ellipsoid.h5"
GEO_SCHEMA = "/repo/simcardems-dev/demos/geometries/lv_ellipsoid.json"

# Generate on rank 0 only, then barrier so all ranks wait before loading
if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
    if not os.path.exists(GEO_PATH):
        mpi_print("Generating LV ellipsoid mesh...")
        cardiac_geometries.create_lv_ellipsoid(
            outdir=".",
            r_short_endo=7.0,
            r_short_epi=10.0,
            r_long_endo=17.0,
            r_long_epi=20.0,
            psize_ref=4.0,
            fiber_angle_endo=-60.0,
            fiber_angle_epi=60.0,
            create_fibers=True,
            fiber_space="Quadrature_3",
        )

dolfin.MPI.comm_world.barrier()
mpi_print("Loading LV ellipsoid geometry...")
from cardiac_geometries.geometry import Geometry

geo = Geometry.from_file(
    fname=GEO_PATH,
    schema_path=GEO_SCHEMA,
    schema=LeftVentricularGeometry.default_schema(),
)

lv_geo = LeftVentricularGeometry.from_geometry(
    geo,
    ep_mesh=geo.mesh,    # same mesh, no refinement
    ffun_ep=geo.ffun,    # same ffun, no adapt needed
)

# # ── Gram-Schmidt orthonormalisation of fibre fields ────────────────────────────
# # P1 nodal fibres lose orthonormality when interpolated to quadrature points
# # independently. Fix by projecting to quadrature space and enforcing
# # orthonormality via Gram-Schmidt before passing to the mechanics solver.
# mpi_print(f"geo info: {geo.info}")
# def gram_schmidt_microstructure(mesh, f0_p1, s0_p1, n0_p1, quadrature_degree=3):
#     element = dolfin.VectorElement(
#         "Quadrature", mesh.ufl_cell(), quadrature_degree,
#         dim=3, quad_scheme="default"
#     )
#     QV = dolfin.FunctionSpace(mesh, element)
#
#     f0_q = dolfin.Function(QV)
#     s0_q = dolfin.Function(QV)
#     n0_q = dolfin.Function(QV)
#
#     f0_q.interpolate(f0_p1)
#     s0_q.interpolate(s0_p1)
#     n0_q.interpolate(n0_p1)
#
#     f_arr = f0_q.vector().get_local().reshape(-1, 3)
#     s_arr = s0_q.vector().get_local().reshape(-1, 3)
#
#     # Gram-Schmidt: normalise f, orthogonalise s against f, n = cross(f, s)
#     f_norm = np.linalg.norm(f_arr, axis=1, keepdims=True)
#     f_arr = f_arr / np.maximum(f_norm, 1e-10)
#
#     s_arr = s_arr - np.sum(s_arr * f_arr, axis=1, keepdims=True) * f_arr
#     s_norm = np.linalg.norm(s_arr, axis=1, keepdims=True)
#     s_arr = s_arr / np.maximum(s_norm, 1e-10)
#
#     n_arr = np.cross(f_arr, s_arr)
#
#     f0_q.vector().set_local(f_arr.flatten())
#     s0_q.vector().set_local(s_arr.flatten())
#     n0_q.vector().set_local(n_arr.flatten())
#     f0_q.vector().apply("insert")
#     s0_q.vector().apply("insert")
#     n0_q.vector().apply("insert")
#
#     # Verify
#     f_dot_s = np.sum(f_arr * s_arr, axis=1)
#     f_dot_n = np.sum(f_arr * n_arr, axis=1)
#     f_norm_check = np.linalg.norm(f_arr, axis=1)
#     if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
#         print(f"After GS — f.s max: {np.abs(f_dot_s).max():.2e}", flush=True)
#         print(f"After GS — f.n max: {np.abs(f_dot_n).max():.2e}", flush=True)
#         print(f"After GS — |f| max: {f_norm_check.max():.6f} min: {f_norm_check.min():.6f}", flush=True)
#
#     return pulse.Microstructure(f0=f0_q, s0=s0_q, n0=n0_q)
#
#
# mpi_print("Orthonormalising fibre fields at quadrature points...")
# ortho_microstructure = gram_schmidt_microstructure(
#     lv_geo.mechanics_mesh,
#     lv_geo.microstructure.f0,
#     lv_geo.microstructure.s0,
#     lv_geo.microstructure.n0,
# )
# lv_geo.microstructure = ortho_microstructure
# lv_geo.microstructure_ep = ortho_microstructure
# # Print first 3 quadrature point fibre values for verification
# f0_arr = lv_geo.microstructure.f0.vector().get_local().reshape(-1, 3)
# s0_arr = lv_geo.microstructure.s0.vector().get_local().reshape(-1, 3)
# n0_arr = lv_geo.microstructure.n0.vector().get_local().reshape(-1, 3)
# mpi_print(f"First 3 f0 quadrature values:\n{f0_arr[:3]}")
# mpi_print(f"First 3 s0 quadrature values:\n{s0_arr[:3]}")
# mpi_print(f"First 3 n0 quadrature values:\n{n0_arr[:3]}")
# ── 2. Whole-mesh uniform activation at end-diastole ──────────────────────────
# interpolate_activation_to_ep_mesh sets non-endo vertices to np.inf
# (never fires). To get every cell firing simultaneously on a coarse mesh
# where propagation is unreliable, we pass ALL mesh vertices as the source
# coords with a uniform activation time — so the nearest-neighbour lookup
# assigns t_end_diastole to every endo vertex, and we separately mark the
# stimulus domain as the whole mesh so non-endo cells also fire.

T_END_DIASTOLE = 100.0  # ms

mpi_print("Building uniform whole-mesh activation times...")
node_coords = lv_geo.ep_mesh.coordinates()          # (N, 3), all vertices
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
# act_fn now has T_END_DIASTOLE on all endo vertices, np.inf on interior/epi.
# Override the np.inf entries to also fire at T_END_DIASTOLE so the
# PrecomputedStimulusUpdater triggers every cell:
local_vec = act_fn.vector().get_local()
local_vec[local_vec == np.inf] = T_END_DIASTOLE
act_fn.vector().set_local(local_vec)
act_fn.vector().apply("insert")
dolfin.MPI.comm_world.barrier()

# Stimulus domain: whole mesh (marker=1 everywhere)
mpi_print("Building whole-mesh stimulus domain...")
tdim = lv_geo.ep_mesh.topology().dim()
cell_domain = dolfin.MeshFunction("size_t", lv_geo.ep_mesh, tdim)
cell_domain.set_all(1)
lv_geo.stimulus_domain = StimulusDomain(domain=cell_domain, marker=1)


# ── 3. Config ──────────────────────────────────────────────────────────────────

mpi_print("Configuring...")
config = Config()
config.T              = 800.0
config.dt      = 1.0 # ms — EP timestep
config.mechanics_solve_strategy = "hybrid"  # or "threshold"
config.dt_mech = 5.0  # ms
config.geometry_path  = GEO_PATH
config.outdir         = "output_lv_ellipsoid"
config.coupling_type  = "fully_coupled_Tor_Land"
# config.coupling_type = "explicit_ORdmm_Land"
config.save_freq      = 20
# config.linear_mechanics_solver = "gmres"
config.spring         = 10.0   # kPa/mm epicardial Robin
config.debug_mode = True
config.mechanics_use_custom_newton_solver = True
# config.mechanics_use_custom_newton_solver = True
# config.linear_mechanics_solver = "gmres"

# ── 4. Build EM coupling ───────────────────────────────────────────────────────
# import simcardems.models.fully_coupled_Tor_Land.active_model as am
# import simcardems.models.fully_coupled_Tor_Land as ftl
#
# OriginalLandModel = am.LandModel
#
# class DebugLandModel(OriginalLandModel):
#     def __init__(self, coupling, parameters=None):
#         mpi_print(f"LandModel receiving f0 element: {coupling.geometry.f0.function_space().ufl_element()}")
#         f0_arr = coupling.geometry.f0.vector().get_local().reshape(-1, 3)
#         s0_arr = coupling.geometry.s0.vector().get_local().reshape(-1, 3)
#         n0_arr = coupling.geometry.n0.vector().get_local().reshape(-1, 3)
#         f_norm = np.linalg.norm(f0_arr, axis=1)
#         s_norm = np.linalg.norm(s0_arr, axis=1)
#         n_norm = np.linalg.norm(n0_arr, axis=1)
#         f_dot_s = np.sum(f0_arr * s0_arr, axis=1)
#         f_dot_n = np.sum(f0_arr * n0_arr, axis=1)
#         mpi_print(f"|f| max: {f_norm.max():.6f} min: {f_norm.min():.6f}")
#         mpi_print(f"|s| max: {s_norm.max():.6f} min: {s_norm.min():.6f}")
#         mpi_print(f"|n| max: {n_norm.max():.6f} min: {n_norm.min():.6f}")
#         mpi_print(f"f.s max: {np.abs(f_dot_s).max():.2e}")
#         mpi_print(f"f.n max: {np.abs(f_dot_n).max():.2e}")
#         mpi_print(f"First 3 f0:\n{f0_arr[:3]}")
#         mpi_print(f"First 3 s0:\n{s0_arr[:3]}")
#         mpi_print(f"First 3 n0:\n{n0_arr[:3]}")
#         super().__init__(coupling, parameters)
# am.LandModel = DebugLandModel
# ftl.ActiveModel = DebugLandModel  # also patch the package-level alias

mpi_print("Setting up EM model...")
# dolfin.set_log_level(dolfin.LogLevel.TRACE)
coupling = em_model.setup_EM_model_from_config(
    config,
    geometry=lv_geo,
    activation_times=act_fn,
)
# dolfin.set_log_level(dolfin.LogLevel.WARNING)  # turn off after setup

# ── 5. LV-only cycle controller ───────────────────────────────────────────────

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
    t_zero=50.0,              # end of preload ramp — 0 to 0.5 kPa
    t_prestress=0.0,
    preload_pressure=0.5,     # kPa at end of preload
    prestress_pressure=0.0,
    t_end_diastole=100.0,     # filling continues to 100 ms
    p_end_diastole=1.0,       # kPa at end of diastole
    gain_contraction=(1.0, 0.5),
    gain_relaxation=(0.5, 0.2),
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
    t_zero=1e9, t_prestress=0.0,
    preload_pressure=0.0, prestress_pressure=0.0,
    t_end_diastole=1e9, p_end_diastole=0.0,
    gain_contraction=(1.0, 0.5), gain_relaxation=(0.5, 0.2),
    p_fill=0.0, period=800.0,
    windkessel=WindkesselParams(p_init=0.0, compliance=1.0, resistance=1.0, evolve=False),
)

lv_state = CavityState(name="LV",      params=lv_params)
rv_state = CavityState(name="RV_stub", params=rv_params)

lv_pressure_const = dolfin.Constant(0.01)
config.traction = 0.01  # dolfin.Constant passes through float_to_constant unchanged

cycle_controller = BiVCycleController(
    lv_state=lv_state,
    rv_state=rv_state,
    lv_pressure_constant=lv_pressure_const,  # same object wired into weak form
    rv_pressure_constant=dolfin.Constant(0.0),
    geometry=lv_geo,
    lv_marker=LV_ENDO_MARKER,
    rv_marker=LV_ENDO_MARKER,
)

u0 = mech_problem.state.split(deepcopy=True)[0]
cycle_controller.initialize(u0)
mpi_print(f"Initial LV volume: {lv_state.volume_n:.2f}")


# ── 6. Time loop ───────────────────────────────────────────────────────────────

mpi_print("Starting time loop...")
os.makedirs(config.outdir, exist_ok=True)

runner = LVCycleRunner.from_models(coupling=coupling, config=config)
runner.set_cycle_controller(cycle_controller, mech_problem, config.outdir)

u0, _ = mech_problem.state.split(deepcopy=True)
cycle_controller.initialize(u0)
mpi_print(f"Initial LV volume: {lv_state.volume_n:.2f}")

runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=True)

runner.save_pv_log()
mpi_print("Done.")

