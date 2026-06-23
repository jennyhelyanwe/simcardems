import os
cache_dir = os.environ.get("FENICS_CACHE_DIR", os.path.expanduser("~/.cache"))
os.environ["XDG_CACHE_HOME"] = cache_dir

import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import dolfin
import numpy as np
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

def mpi_print(*args, **kwargs):
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print(*args, **kwargs, flush=True)


# ── BiVCycleRunner ─────────────────────────────────────────────────────────────

class BiVCycleRunner(Runner):
    def set_cycle_controller(self, controller, mech_problem, outdir, pseudo_ecg=None, dt_ecg=5.0):
        self._cycle_controller = controller
        self._mech_problem = mech_problem
        self._outdir = outdir
        self._pv_log = []
        self._pseudo_ecg = pseudo_ecg
        self._dt_ecg = dt_ecg
        self._last_ecg_t = -dt_ecg  # ensure first evaluation happens

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
        self._pv_log.append((
            t_ms,
            lv.pressure_n, lv.volume_n,
            rv.pressure_n, rv.volume_n,
        ))
        if self._pseudo_ecg is not None:
            t_ms = TimeStepper.ns2ms(self.t)
            if t_ms - self._last_ecg_t >= self._dt_ecg - 1e-10:
                self.coupling.assigners.assign_ep()
                v_fn = self.coupling.assigners.functions["ep"]["V"]
                self._pseudo_ecg.compute(v_fn, t_ms)
                self._last_ecg_t = t_ms

    def save_pv_log(self):
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            arr = np.array(self._pv_log)
            path = os.path.join(self._outdir, "pv_loop.csv")
            np.savetxt(
                path, arr, delimiter=",",
                header="t_ms,LVP_kPa,LVV,RVP_kPa,RVV", comments="",
            )
            mpi_print(f"PV loop saved to {path}")

# Set up Pseudo ECG
from simcardems.postprocess import ecg_recovery
class PseudoECG:
    """
    Computes pseudo-ECG leads at user-defined electrode locations
    using the lead field / reciprocity formula at each mechanics timestep.

    Electrode locations in mm, same coordinate system as the mesh.
    Lead = phi(positive_electrode) - phi(negative_electrode)
    """

    def __init__(self, leads: dict, sigma_b: float = 1.0):
        """
        leads: dict of lead_name -> (positive_electrode, negative_electrode)
               each electrode is a (x, y, z) tuple in mm
        sigma_b: bulk conductivity in mS/mm, default 1.0
        """
        self.leads = leads
        self.sigma_b = sigma_b
        self.log = {name: [] for name in leads}
        self.times = []

    def compute(self, v: dolfin.Function, t: float):
        """Call at each timestep where V is available."""
        mesh = v.function_space().mesh()
        for name, (pos, neg) in self.leads.items():
            phi_pos = ecg_recovery(
                v=v, sigma_b=self.sigma_b,
                point=np.array(pos), mesh=mesh,
            )
            phi_neg = ecg_recovery(
                v=v, sigma_b=self.sigma_b,
                point=np.array(neg), mesh=mesh,
            )
            self.log[name].append(phi_pos - phi_neg)
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
import pandas as pd
node_coords = pd.read_csv(
    "./rodero_05_fine/rodero_05_fine_xyz.csv", header=None
).to_numpy() * 10.0

path = "./heart.endocardial-activation-times"
node_ids, activation_times, coords_subset = load_activation_times(path, node_coords)
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
config.T                              = 800.0   # ms — one full beat
config.dt                             = 0.05    # ms — EP timestep
config.dt_mech                        = 1.0     # ms — mechanics timestep
config.geometry_path                  = "rodero_05_fine.h5"
config.outdir                         = "biv_run_output"
config.coupling_type                  = "fully_coupled_Tor_Land"
config.save_freq                      = 20      # save every 20 mechanics steps = every 20 ms
config.linear_mechanics_solver        = "mumps"
config.spring                         = 10.0    # kPa/mm epicardial Robin
config.traction                       = 0.01    # kPa — ensures Neumann BCs created
config.mechanics_use_custom_newton_solver = True
config.mechanics_solve_strategy       = "hybrid"


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


# ── 7. Cycle controller ────────────────────────────────────────────────────────

LV_ENDO_MARKER = biv_geo.markers["ENDO_LV"][0]
RV_ENDO_MARKER = biv_geo.markers["ENDO_RV"][0]

mech_problem = coupling.mech_solver

# Fish out Neumann BC constants
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

# LV params — preload 0→0.5 kPa over 50 ms, fill to 1.0 kPa by 100 ms
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
    preload_pressure=0.17,       # 0.5 / 3
    prestress_pressure=0.0,
    t_end_diastole=100.0,
    p_end_diastole=0.33,         # 1.0 / 3
    gain_contraction=(1.0, 0.5),
    gain_relaxation=(0.5, 0.2),
    p_fill=0.033,                # 0.1 / 3
    period=800.0,
    windkessel=WindkesselParams(
        p_init=3.0,              # ~9.0 / 3
        compliance=3.0,          # higher compliance than LV
        resistance=33.0,         # 100.0 / 3
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


# ── 8. Run ─────────────────────────────────────────────────────────────────────
# Electrode locations in mm — adjust to your mesh coordinate system
electrode_df = pd.read_csv(
    "./rodero_05_fine/rodero_05_fine_nodefield_electrode_xyz.csv",
    header=None, names=["x", "y", "z"],
)
electrodes = electrode_df.to_numpy()  # mm

LA, RA, LL = electrodes[0], electrodes[1], electrodes[2]
V1, V2, V3, V4, V5, V6 = electrodes[4], electrodes[5], electrodes[6], electrodes[7], electrodes[8], electrodes[9]

# Wilson central terminal
wct = tuple((LA + RA + LL) / 3.0)

pseudo_ecg = PseudoECG(
    leads={
        # Limb leads
        "I":    (tuple(LA), tuple(RA)),
        "II":   (tuple(LL), tuple(RA)),
        "III":  (tuple(LL), tuple(LA)),
        # Augmented limb leads
        "aVR":  (tuple(RA), tuple((LA + LL) / 2.0)),
        "aVL":  (tuple(LA), tuple((RA + LL) / 2.0)),
        "aVF":  (tuple(LL), tuple((RA + LA) / 2.0)),
        # Precordial leads (unipolar vs WCT)
        "V1":   (tuple(V1), wct),
        "V2":   (tuple(V2), wct),
        "V3":   (tuple(V3), wct),
        "V4":   (tuple(V4), wct),
        "V5":   (tuple(V5), wct),
        "V6":   (tuple(V6), wct),
    },
    sigma_b=1.0,
)

os.makedirs(config.outdir, exist_ok=True)
runner = BiVCycleRunner.from_models(coupling=coupling, config=config)
runner.set_cycle_controller(
    cycle_controller, mech_problem, config.outdir,
    pseudo_ecg=pseudo_ecg, dt_ecg=5.0,
)
runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=True)
runner.save_pv_log()
mpi_print("Done.")