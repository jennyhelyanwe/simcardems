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

geo = Geometry.from_file("rodero_05_coarse.h5")
biv_geo = BiVentricularGeometry.from_geometry(
    geo,
    ep_mesh=geo.mesh,
    ffun_ep=geo.ffun,
)
mpi_print(f"Mesh vertices: {biv_geo.ep_mesh.num_vertices()}")
# import numpy as np
# f0_arr = biv_geo.microstructure.f0.vector().get_local().reshape(-1, 3)
# s0_arr = biv_geo.microstructure.s0.vector().get_local().reshape(-1, 3)
# n0_arr = biv_geo.microstructure.n0.vector().get_local().reshape(-1, 3)
# mpi_print(f"|f0| min={np.linalg.norm(f0_arr, axis=1).min():.6f} max={np.linalg.norm(f0_arr, axis=1).max():.6f}")
# mpi_print(f"|s0| min={np.linalg.norm(s0_arr, axis=1).min():.6f} max={np.linalg.norm(s0_arr, axis=1).max():.6f}")
# mpi_print(f"|n0| min={np.linalg.norm(n0_arr, axis=1).min():.6f} max={np.linalg.norm(n0_arr, axis=1).max():.6f}")
# quit()

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
config.geometry_path                      = "rodero_05_coarse.h5"
config.outdir                             = "biv_coarse_run_output"
config.coupling_type                      = "fully_coupled_Tor_Land"
config.save_freq                          = 20
config.linear_mechanics_solver            = "mumps"
config.spring                             = 10.0
config.traction                           = 0.001
config.mechanics_use_custom_newton_solver = True
config.mechanics_solve_strategy           = "hybrid"
config.mech_threshold                     = 1.0
config.relaxation_factor                  = 0.3
# config.set_material                       = "Guccione"


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
# import simcardems.boundary_conditions as _bcs
# _original_create_biv = _bcs.create_biv_boundary_conditions
#
# def _patched_create_biv(geo, traction_lv=None, traction_rv=None, spring=None):
#     from functools import partial
#     import dolfin
#     import pulse
#
#     lv_pressure = dolfin.Constant(0.0) # _bcs.utils.float_to_constant(traction_lv) if traction_lv is not None else dolfin.Constant(0.0)
#     rv_pressure = dolfin.Constant(0.0)  # RV fixed at zero
#
#     neumann_bc = [
#         pulse.NeumannBC(traction=lv_pressure, marker=geo.markers["ENDO_LV"][0]),
#         pulse.NeumannBC(traction=rv_pressure, marker=geo.markers["ENDO_RV"][0]),
#     ]
#
#     dirichlet_bc = [
#         partial(
#             lambda W, ffun, marker: dolfin.DirichletBC(
#                 W.sub(0), dolfin.Constant((0, 0, 0)), ffun, marker
#             ),
#             ffun=geo.ffun,
#             marker=geo.markers["EPI"][0],
#         ),
#         partial(
#             lambda W, ffun, marker: dolfin.DirichletBC(
#                 W.sub(0), dolfin.Constant((0, 0, 0)), ffun, marker
#             ),
#             ffun=geo.ffun,
#             marker=geo.markers["BASE"][0],
#         ),
#     ]
#
#     return pulse.BoundaryConditions(
#         dirichlet=dirichlet_bc,
#         neumann=neumann_bc,
#         robin=[],
#     )
#
# _bcs.create_biv_boundary_conditions = _patched_create_biv
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
config.traction = 0.0 #  float(lv_pressure_const)

runner = BiVCycleRunner.from_models(coupling=coupling, config=config)
runner.set_cycle_controller(
    cycle_controller, mech_problem, config.outdir,
    pseudo_ecg=pseudo_ecg, dt_ecg=5.0,
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

lv_pressure_const.assign(0.02)
rv_pressure_const.assign(0.007)
b = dolfin.assemble(mech_problem._virtual_work)
x = mech_problem.state.vector()
mpi_print(f"Residual before BC apply: {b.norm('l2'):.6e}")
for bc in mech_problem._dirichlet_bc:
    bc.apply(b, x)
mpi_print(f"Residual after BC apply: {b.norm('l2'):.6e}")
dolfin.PETScOptions.set("mat_mumps_icntl_4", "3")
dolfin.PETScOptions.set("mat_mumps_icntl_14", "500")
lv_pressure_const.assign(0.0)
rv_pressure_const.assign(0.0)
try:
    runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=True)
except Exception as e:
    mpi_print(f"Runner error: {e}")
finally:
    runner.close_files()
    mpi_print("Done.")