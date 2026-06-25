import os
cache_dir = os.environ.get("FENICS_CACHE_DIR", os.path.expanduser("~/.cache"))
os.environ["XDG_CACHE_HOME"] = cache_dir

import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("simcardems.newton_solver").setLevel(logging.DEBUG)
logging.getLogger("simcardems.biv_cavity_cycle_controller").setLevel(logging.DEBUG)

import dolfin
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
                              pseudo_ecg=None, dt_ecg=5.0):
        self._cycle_controller = controller
        self._mech_problem     = mech_problem
        self._outdir           = outdir
        self._pseudo_ecg       = pseudo_ecg
        self._dt_ecg           = dt_ecg
        self._last_ecg_t       = -dt_ecg

        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            self._pv_file = open(os.path.join(outdir, "pv_loop.csv"), "w")
            self._pv_file.write("t_ms,LVP_kPa,LVV,RVP_kPa,RVV\n")
            self._pv_file.flush()
        else:
            self._pv_file = None

        if pseudo_ecg is not None:
            pseudo_ecg.open_file(os.path.join(outdir, "pseudo_ecg.csv"))

    def _solve_mechanics(self):
        self.coupling.coupling_to_mechanics()
        self.coupling.solve_mechanics()
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
            f"  LV phase={lv.phase}  LVP={lv.pressure_n:.3f} kPa  LVV={lv.volume_n:.2f}"
            f"  RV phase={rv.phase}  RVP={rv.pressure_n:.3f} kPa  RVV={rv.volume_n:.2f}"
        )

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


# ── 1. Geometry ────────────────────────────────────────────────────────────────
mpi_print('Build geometry...')
from cardiac_geometries.geometry import Geometry
from simcardems.bivgeometry import BiVentricularGeometry

geo = Geometry.from_file("rodero_05_fine.h5")
biv_geo = BiVentricularGeometry.from_geometry(
    geo,
    ep_mesh=geo.mesh,
    ffun_ep=geo.ffun,
)
mpi_print(f"Mesh vertices: {biv_geo.ep_mesh.num_vertices()}")

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
config.dt                                 = 0.05
config.dt_mech                            = 1.0
config.geometry_path                      = "rodero_05_fine.h5"
config.outdir                             = "biv_run_output"
config.coupling_type                      = "fully_coupled_Tor_Land"
config.save_freq                          = 20
config.linear_mechanics_solver            = "gmres"
config.spring                             = 10.0
# config.traction                           = 0.01
config.traction                           = 0.001
# config.traction                           = 0.5
config.mechanics_use_custom_newton_solver = False
config.mechanics_solve_strategy           = "hybrid"


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
# import simcardems.mechanics_model as mm
# original_solve = mm.MechanicsProblem.solve
#
# def patched_solve(self):
#     rank = dolfin.MPI.rank(dolfin.MPI.comm_world)
#
#     self._init_forms(init_solver=False)
#
#     import petsc4py
#     petsc4py.init()
#     from petsc4py import PETSc
#
#     A = dolfin.PETScMatrix()
#     dolfin.assemble(self._jacobian, tensor=A)
#
#     norm_frob = A.norm('frobenius')
#     norm_inf = A.norm('linf')
#     norm_1 = A.norm('l1')
#     nnz = A.nnz()
#
#     if rank == 0:
#         print(f"Jacobian size: {A.size(0)} x {A.size(1)}", flush=True)
#         print(f"Jacobian NNZ: {nnz}", flush=True)
#         print(f"Jacobian Frobenius norm: {norm_frob:.6e}", flush=True)
#         print(f"Jacobian linf norm (max row sum): {norm_inf:.6e}", flush=True)
#         print(f"Jacobian l1 norm (max col sum): {norm_1:.6e}", flush=True)
#
#     dolfin.PETScOptions.clear()
#     dolfin.PETScOptions.set("ksp_type", "preonly")
#     dolfin.PETScOptions.set("pc_type", "lu")
#     dolfin.PETScOptions.set("pc_factor_mat_solver_type", "mumps")
#     dolfin.PETScOptions.set("mat_mumps_icntl_4", "1")   # errors only, no stats dump
#     self.solver.linear_solver().set_from_options()
#     return original_solve(self)
#
# # ─────────────────────────────────────────────────────────────────
#
# mm.MechanicsProblem.solve = patched_solve
# dolfin.PETScOptions.set("mat_mumps_icntl_14", "200")
coupling = em_model.setup_EM_model_from_config(
    config,
    geometry=biv_geo,
    activation_times=act_fn,
    celltype_function=cell_fn,
    iks_scale_function=iks_fn,
)


# ── 7. Cycle controller ────────────────────────────────────────────────────────

LV_ENDO_MARKER = biv_geo.markers["ENDO_LV"][0]
RV_ENDO_MARKER = biv_geo.markers["ENDO_RV"][0]
mech_problem   = coupling.mech_solver

lv_pressure_const = None
rv_pressure_const = None
for nbc in mech_problem.bcs.neumann:
    if nbc.marker == LV_ENDO_MARKER:
        lv_pressure_const = nbc.traction
    elif nbc.marker == RV_ENDO_MARKER:
        rv_pressure_const = nbc.traction

if lv_pressure_const is None:
    raise RuntimeError("No Neumann BC found on ENDO_LV")
if rv_pressure_const is None:
    raise RuntimeError("No Neumann BC found on ENDO_RV")

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
    t_zero=50.0,
    t_prestress=0.0,
    preload_pressure=0.17,
    prestress_pressure=0.0,
    t_end_diastole=100.0,
    p_end_diastole=0.33,
    gain_contraction=(1.0, 0.5),
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
config.traction = float(lv_pressure_const)  # reset to float for HDF5 serialisation

runner = BiVCycleRunner.from_models(coupling=coupling, config=config)
runner.set_cycle_controller(
    cycle_controller, mech_problem, config.outdir,
    pseudo_ecg=pseudo_ecg, dt_ecg=5.0,
)

try:
    runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=True)
except Exception as e:
    mpi_print(f"Runner error: {e}")
finally:
    runner.close_files()
    mpi_print("Done.")