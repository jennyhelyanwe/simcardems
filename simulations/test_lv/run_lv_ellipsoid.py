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
    def set_cycle_controller(self, controller, mech_problem, outdir, pseudo_ecg=None, dt_ecg=5.0):
        self._cycle_controller = controller
        self._mech_problem = mech_problem
        self._outdir = outdir
        self._pseudo_ecg = pseudo_ecg
        self._dt_ecg = dt_ecg
        self._last_ecg_t = -dt_ecg

        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            self._pv_file = open(os.path.join(outdir, "pv_loop.csv"), "w")
            self._pv_file.write("t_ms,LVP_kPa,LVV\n")
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
            self._pv_file.write(
                f"{t_ms:.3f},{state.pressure_n:.6f},{state.volume_n:.6f}\n"
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

    def save_pv_log(self):
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            pv_arr = np.array(self._pv_log)
            path = os.path.join(self._outdir, "pv_loop.csv")
            np.savetxt(path, pv_arr, delimiter=",",
                       header="t_ms,LVP_kPa,LVV", comments="")
            mpi_print(f"PV loop saved to {path}")


# ── 1. Generate / load LV ellipsoid geometry ───────────────────────────────────

GEO_PATH   = "/repo/simcardems-dev/demos/geometries/lv_ellipsoid.h5"
GEO_SCHEMA = "/repo/simcardems-dev/demos/geometries/lv_ellipsoid.json"

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
    ep_mesh=geo.mesh,
    ffun_ep=geo.ffun,
)

from simcardems.postprocess import ecg_recovery


class PseudoECG:
    """
    Computes pseudo-ECG leads at user-defined electrode locations
    using the lead field / reciprocity formula at each mechanics timestep.

    Electrode locations in mm, same coordinate system as the mesh.
    Lead = phi(positive_electrode) - phi(negative_electrode)
    """

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

    def save(self, path: str):
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            header = "t_ms," + ",".join(self.leads.keys())
            data = np.column_stack([
                self.times,
                *[self.log[name] for name in self.leads]
            ])
            np.savetxt(path, data, delimiter=",", header=header, comments="")
            mpi_print(f"Pseudo-ECG saved to {path}")


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

# ── 2. Whole-mesh uniform activation at end-diastole ──────────────────────────
# interpolate_activation_to_ep_mesh sets non-endo vertices to np.inf
# (never fires). To get every cell firing simultaneously on a coarse mesh
# where propagation is unreliable, we pass ALL mesh vertices as the source
# coords with a uniform activation time — so the nearest-neighbour lookup
# assigns t_end_diastole to every endo vertex, and we separately mark the
# stimulus domain as the whole mesh so non-endo cells also fire.

T_END_DIASTOLE = 100.0  # ms

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

# ── 3. Config ──────────────────────────────────────────────────────────────────

mpi_print("Configuring...")
config = Config()
config.T              = 800.0
config.dt      = 1.0
config.mechanics_solve_strategy = "hybrid"
config.dt_mech                  = 5.0
config.mech_threshold           = 1.0
config.geometry_path  = GEO_PATH
config.outdir         = "output_lv_ellipsoid"
config.coupling_type  = "fully_coupled_Tor_Land"
config.save_freq      = 20
config.linear_mechanics_solver = "mumps"
config.spring         = 10.0
config.debug_mode = True
config.mechanics_use_custom_newton_solver = False

# ── 4. Build EM coupling ───────────────────────────────────────────────────────

mpi_print("Setting up EM model...")
coupling = em_model.setup_EM_model_from_config(
    config,
    geometry=lv_geo,
    activation_times=act_fn,
)

# ── 5. LV-only cycle controller ───────────────────────────────────────────────

LV_ENDO_MARKER = lv_geo.markers["ENDO"][0]

mech_problem = coupling.mech_solver
lv_pressure_const = None
for nbc in mech_problem.bcs.neumann:
    if nbc.marker == LV_ENDO_MARKER:
        lv_pressure_const = nbc.traction  # the actual Constant compiled into the weak form
        break

if lv_pressure_const is None:
    raise RuntimeError("No Neumann BC found on ENDO")

rv_pressure_const = dolfin.Constant(0.0)

lv_params = CycleParams(
    t_zero=50.0,
    t_prestress=0.0,
    preload_pressure=0.5,
    prestress_pressure=0.0,
    t_end_diastole=100.0,
    p_end_diastole=1.0,
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

# NOTE: previously this block overwrote lv_pressure_const with a brand-new,
# disconnected dolfin.Constant(0.01), which meant the cycle controller's
# pressure updates never reached the actual assembled weak form (nbc.traction
# stayed frozen). That overwrite has been removed — lv_pressure_const now
# stays bound to the same Constant object referenced in mech_problem.bcs.neumann.

cycle_controller = BiVCycleController(
    lv_state=lv_state,
    rv_state=rv_state,
    lv_pressure_constant=lv_pressure_const,
    rv_pressure_constant=rv_pressure_const,
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


def close_files(self):
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        if self._pv_file:
            self._pv_file.close()
        if self._ecg_file:
            self._ecg_file.close()


runner = LVCycleRunner.from_models(coupling=coupling, config=config)
runner.set_cycle_controller(cycle_controller, mech_problem, config.outdir, pseudo_ecg=pseudo_ecg, dt_ecg=5.0)

u0, _ = mech_problem.state.split(deepcopy=True)
cycle_controller.initialize(u0)
mpi_print(f"Initial LV volume: {lv_state.volume_n:.2f}")

try:
    runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=True)
except Exception as e:
    mpi_print(f"Runner crashed: {e}")
finally:
    runner.close_files()
    mpi_print("Done.")