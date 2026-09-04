"""
run_biv_circulation.py
------------------------
BiV mesh coupled to a genuine 3D EP model (spatially-varying, live Ta) and
Henrik Finsberg's `circulation` package for closed-loop Regazzoni 0D
circulation. BOTH cavities are volume-constrained (Lagrange multiplier)
PERMANENTLY — no phase detection, no mode-switching, no Ta-ramping. Every
call is: sub-loop EP at fine resolution, ONE mechanics solve, full
bookkeeping update. This mirrors the only pattern that's been reliably
crash-free throughout debugging (a single solve + bookkeeping, never two
solves on the same problem object without a bookkeeping update between).

Isovolumic behavior emerges from the 0D model's own valve resistance law
(near-infinite when closed) — no explicit phase machinery needed on the
3D side at all.
"""

import os

cache_dir = os.environ.get("FENICS_CACHE_DIR", os.path.expanduser("~/.cache"))
os.environ["XDG_CACHE_HOME"] = cache_dir

import logging
logging.getLogger("simcardems.newton_solver").setLevel(logging.DEBUG)
logging.getLogger("__main__").setLevel(logging.DEBUG)

import time
import numpy as np
import pandas as pd

import dolfin
dolfin.PETScOptions.set("mat_mumps_icntl_4", "0")
dolfin.PETScOptions.set("snes_monitor")

import pulse
from pulse import kinematics as _kinematics
from pulse.dolfin_utils import list_sum as _list_sum
import pulse.mechanicsproblem as _mp
try:
    import ufl_legacy as _ufl
except ImportError:
    import ufl as _ufl

import circulation
from simcardems.postprocess import ecg_recovery
import logging
if dolfin.MPI.rank(dolfin.MPI.comm_world) != 0:
    logging.getLogger("circulation").setLevel(logging.WARNING)

from cardiac_geometries.geometry import Geometry
from simcardems.bivgeometry import BiVentricularGeometry
from simcardems.activation import load_activation_times, interpolate_activation_to_ep_mesh, endocardial_stimulus_domain
from simcardems.spatial_fields import map_dense_field_to_dg0_function, load_dense_node_field, map_dense_field_to_ep_mesh
from simcardems.config import Config
from simcardems.models import em_model
from simcardems.models.fully_coupled_Tor_Land.cell_model import TorLandFull
from simcardems.geometry import refine_mesh, StimulusDomain
from simcardems.steady_state_cell_cache import load_steady_state_cache, apply_celltype_aware_initial_conditions
from simcardems import utils

t_script_start = time.time()
logger = utils.getLogger(__name__)
logger.info(['dolfin:', dolfin.__version__])

# ── Run-mode toggle ──────────────────────────────────────────────────────
RUN_MODE = os.environ.get("RUN_MODE", "local")
assert RUN_MODE in ("local", "archer2")
NUM_REFINEMENTS = 3 if RUN_MODE == "archer2" else 0
UNIFORM_ACTIVATION_TEST = (RUN_MODE == "local")
logger.info(f"RUN_MODE = {RUN_MODE} (NUM_REFINEMENTS={NUM_REFINEMENTS}, UNIFORM_ACTIVATION_TEST={UNIFORM_ACTIVATION_TEST})")

RESOLUTION = "4mm"
MESH_DIR = "meshes/"
RUN_TAG = os.environ.get("RUN_TAG", "circulation_default")
RESULTS_DIR = f"results_{RESOLUTION}/{RUN_TAG}/"
os.makedirs(RESULTS_DIR, exist_ok=True)

EP_SUBSTEP_DT = 0.05  # ms — matches your existing fine EP resolution
CIRC_DT = 1.0         # ms — circulation model's own outer timestep
NUM_BEATS = 3

class PseudoECG:
    def __init__(self, leads: dict, sigma_b: float = 1.0):
        self.leads = leads
        self.sigma_b = sigma_b
        self._file = None

    def open_file(self, path):
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            self._file = open(path, "w")
            self._file.write("t_ms," + ",".join(self.leads.keys()) + "\n")
            self._file.flush()

    def compute(self, v, t):
        mesh = v.function_space().mesh()
        vals = []
        for name, (pos, neg) in self.leads.items():
            phi_pos = ecg_recovery(v=v, sigma_b=self.sigma_b, point=np.array(pos), mesh=mesh)
            phi_neg = ecg_recovery(v=v, sigma_b=self.sigma_b, point=np.array(neg), mesh=mesh)
            vals.append(phi_pos - phi_neg)
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0 and self._file is not None:
            self._file.write(f"{t:.3f}," + ",".join(f"{v:.8f}" for v in vals) + "\n")
            self._file.flush()

    def close(self):
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0 and self._file is not None:
            self._file.close()

# ── Monkey-patch: normal-only Robin BC (frictionless contact) ───────────
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
        r = robin.value * u_normal
        external_work.append(dolfin.inner(r, v) * ds(robin.marker))
    for body_force in self.bcs.body_force:
        external_work.append(-dolfin.derivative(dolfin.inner(body_force, u) * dx, u, v))
    return _list_sum(external_work) if external_work else None


_mp.MechanicsProblem._external_work = _external_work_normal_robin
logger.info("Patched MechanicsProblem._external_work: Robin BC normal-only")


# ── CavityConstrainedMechanicsProblem — ALWAYS both LV+RV constrained ────
class CavityConstrainedMechanicsProblem(pulse.MechanicsProblem):
    def __init__(self, *args, cavity_specs=None, **kwargs):
        if not cavity_specs:
            raise ValueError("cavity_specs must have at least one {marker: v_target}")
        self.cavity_specs = cavity_specs
        self._u_standalone = None
        self._u_assigner = None
        super().__init__(*args, **kwargs)

    def _init_spaces(self):
        mesh = self.geometry.mesh
        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        R_elements = [dolfin.FiniteElement("Real", mesh.ufl_cell(), 0) for _ in self.cavity_specs]
        self.state_space = dolfin.FunctionSpace(mesh, dolfin.MixedElement([P2, P1] + R_elements))
        self.state = dolfin.Function(self.state_space, name="state")
        self.state_test = dolfin.TestFunction(self.state_space)

    def _init_forms(self):
        parts = dolfin.split(self.state)
        test_parts = dolfin.split(self.state_test)
        u, p = parts[0], parts[1]
        v, q = test_parts[0], test_parts[1]
        P_cavities = parts[2:]
        markers = list(self.cavity_specs.keys())

        F = dolfin.variable(_kinematics.DeformationGradient(u))
        J = _kinematics.Jacobian(F)
        mesh = self.geometry.mesh
        dx = self.geometry.dx
        ds = self.geometry.ds
        N = self.geometry.facet_normal

        internal_energy = self.material.strain_energy(F) + self.material.compressibility(p, J)
        self._virtual_work = dolfin.derivative(internal_energy * dx, self.state, self.state_test)

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
        n_weighted = _ufl.cofac(F) * N
        cavity_integrand = -dolfin.dot(x, n_weighted) / 3.0
        total_ref_volume = dolfin.assemble(dolfin.Constant(1.0) * dx)

        cavity_energy = 0
        for marker, P_c in zip(markers, P_cavities):
            v_target = self.cavity_specs[marker]
            cavity_energy += -P_c * cavity_integrand * ds(marker)
            cavity_energy += (P_c * v_target / total_ref_volume) * dx
        self._virtual_work += dolfin.derivative(cavity_energy, self.state, self.state_test)

        self._dirichlet_bc = []
        self._set_dirichlet_bc()
        self._jacobian = dolfin.derivative(self._virtual_work, self.state, dolfin.TrialFunction(self.state_space))
        self._init_solver()

    @property
    def strong_coupling(self):
        return hasattr(self.material.active, "Ta")

    def solve(self):
        nliter, nlconv = self._raw_solve()
        self._update_active_stress_bookkeeping()
        return nliter, nlconv

    def _raw_solve(self):
        return super().solve()

    def _update_active_stress_bookkeeping(self):
        if not self.strong_coupling:
            return
        active = self.material.active
        u = get_u_view(self)
        F = dolfin.grad(u) + dolfin.Identity(3)
        f = F * active.f0
        lmbda = dolfin.sqrt(f ** 2)
        active._projector.project(active.lmbda, lmbda)
        if active.dt > 0:
            active._projector.project(active._dLambda, (lmbda - active.lmbda_prev) / active.dt)

        active._projector.project(active.Ta_current, active.Ta(lmbda))
        ta_arr = active.Ta_current.vector().get_local()
        ta_arr[ta_arr < 0.0] = 0.0  # projection-ringing clamp — see earlier debugging notes
        valve_mask = getattr(self.geometry, "valve_mask", None)
        if valve_mask is not None:
            mask_arr = valve_mask.vector().get_local()
            ta_arr[mask_arr > 0.5] = 0.0
        active.Ta_current.vector().set_local(ta_arr)
        active.Ta_current.vector().apply("insert")

        active.update_current(lmbda=lmbda)
        active.update_prev()

    def get_cavity_pressure(self, marker):
        idx = list(self.cavity_specs.keys()).index(marker)
        sub_idx = 2 + idx
        if not hasattr(self, '_p_standalone'):
            self._p_standalone, self._p_assigner = {}, {}
        if idx not in self._p_standalone:
            V_full = self.state_space
            R = V_full.ufl_element().sub_elements()[sub_idx]
            V_r = dolfin.FunctionSpace(V_full.mesh(), R)
            self._p_standalone[idx] = dolfin.Function(V_r)
            self._p_assigner[idx] = dolfin.FunctionAssigner(V_r, V_full.sub(sub_idx))
        self._p_assigner[idx].assign(self._p_standalone[idx], self.state.sub(sub_idx))
        local = self._p_standalone[idx].vector().get_local()
        local_val = float(local[0]) if len(local) > 0 else 0.0
        return dolfin.MPI.sum(self.geometry.mesh.mpi_comm(), local_val)


def get_u_view(problem):
    if getattr(problem, '_u_standalone', None) is None:
        V_full = problem.state_space
        P2 = V_full.ufl_element().sub_elements()[0]
        V_u = dolfin.FunctionSpace(V_full.mesh(), P2)
        problem._u_standalone = dolfin.Function(V_u)
        problem._u_assigner = dolfin.FunctionAssigner(V_u, V_full.sub(0))
    problem._u_assigner.assign(problem._u_standalone, problem.state.sub(0))
    return problem._u_standalone


# ── 1. Geometry ───────────────────────────────────────────────────────────
logger.info('Build geometry...')
geo = Geometry.from_file(MESH_DIR + "rodero_05_coarse_" + RESOLUTION + ".h5")
if NUM_REFINEMENTS > 1:
    ep_mesh = refine_mesh(geo.mesh, num_refinements=NUM_REFINEMENTS)
    ffun_ep = dolfin.adapt(geo.ffun, ep_mesh)
    biv_geo = BiVentricularGeometry.from_geometry(geo, ep_mesh=ep_mesh, ffun_ep=ffun_ep,
                                                    parameters={"num_refinements": NUM_REFINEMENTS})
else:
    biv_geo = BiVentricularGeometry.from_geometry(geo, ep_mesh=geo.mesh, ffun_ep=geo.ffun)

coarse_tv = np.load(MESH_DIR + '/rodero_05_coarse_' + RESOLUTION + '_tv.npy')
is_valve_float = (coarse_tv >= 7).astype(float)
coarse_centres_all = np.array([cell.midpoint().array() for cell in dolfin.cells(biv_geo.mechanics_mesh)])
valve_fn = map_dense_field_to_ep_mesh(biv_geo.mechanics_mesh, coarse_centres_all, is_valve_float)
biv_geo.valve_mask = valve_fn

# ── 2. Activation times ───────────────────────────────────────────────────
node_coords = pd.read_csv(MESH_DIR + "/rodero_05_fine_xyz.csv", header=None).to_numpy() * 10.0
node_ids, activation_times, coords_subset = load_activation_times(MESH_DIR + "/heart.endocardial-activation-times", node_coords)
activation_times = activation_times * 1000.0

act_fn = interpolate_activation_to_ep_mesh(
    ep_mesh=biv_geo.ep_mesh,
    endo_marker_ep=[biv_geo.markers["ENDO_LV"][0], biv_geo.markers["ENDO_RV"][0]],
    ffun_ep=biv_geo.ffun_ep, node_ids_mech=node_ids,
    activation_times_mech=activation_times, coords_subset_mech=coords_subset,
)

if UNIFORM_ACTIVATION_TEST:
    act_fn.vector()[:] = 0.0
    whole_mesh_marker = dolfin.MeshFunction("size_t", biv_geo.ep_mesh, biv_geo.ep_mesh.topology().dim(), 1)
    biv_geo.stimulus_domain = StimulusDomain(domain=whole_mesh_marker, marker=1)
else:
    biv_geo.stimulus_domain = endocardial_stimulus_domain(
        mesh=biv_geo.ep_mesh, ffun=biv_geo.ffun_ep,
        endo_markers=[biv_geo.markers["ENDO_LV"][0], biv_geo.markers["ENDO_RV"][0]], layer_thickness=2,
    )

# ── Steady-state cell model cache ─────────────────────────────────────────
CELL_PARAMS_OVERRIDE = {}
PCL = 800.0
steady_state = load_steady_state_cache(CELL_PARAMS_OVERRIDE, PCL, max_beats=300)
cell_init_file = steady_state["cell_init_files"][0]

# ── 3. Config ──────────────────────────────────────────────────────────────
config = Config()
config.cell_init_file = cell_init_file
config.T = NUM_BEATS * 800.0  # nominal only — circulation model owns the real loop
config.dt = EP_SUBSTEP_DT
config.dt_mech = CIRC_DT  # nominal — mechanics now solves once per p_BiV_func call, not gated by this
config.geometry_path = MESH_DIR + "rodero_05_coarse_" + RESOLUTION + ".h5"
config.outdir = RESULTS_DIR + "biv_coarse_run_output"
config.coupling_type = "fully_coupled_Tor_Land"
config.save_freq = 10
config.linear_mechanics_solver = "mumps"
config.spring = 50.0
config.traction = 0.0
config.mechanics_use_custom_newton_solver = True
config.mechanics_solve_strategy = "fixed"

material_params_override = dict(a=2.28, a_f=1.686, b=9.726, b_f=15.779, a_s=0.0, b_s=0.0, a_fs=0.0, b_fs=0.0)

# ── 4. Spatial fields ────────────────────────────────────────────────────
ct_values = load_dense_node_field(MESH_DIR + "rodero_05_fine_nodefield_cell-type.csv")
cell_fn = map_dense_field_to_dg0_function(biv_geo.ep_mesh, node_coords, ct_values, {1: 0, 2: 2, 3: 1})
iks_values = load_dense_node_field(MESH_DIR + "rodero_05_fine_nodefield_sf_IKs.csv")
iks_fn = map_dense_field_to_ep_mesh(biv_geo.ep_mesh, node_coords, iks_values)

# ── 5. EM coupling ────────────────────────────────────────────────────────
logger.info('Setting up EM model...')
coupling = em_model.setup_EM_model_from_config(
    config, geometry=biv_geo, activation_times=act_fn,
    celltype_function=cell_fn, iks_scale_function=iks_fn,
    material_parameters=material_params_override,
    valve_stiffness_scale=3.0,
)
mech_problem_base = coupling.mech_solver  # only used for .material / .bcs.robin now

state_names = list(TorLandFull.default_initial_conditions().keys())
apply_celltype_aware_initial_conditions(coupling, cell_fn, steady_state["cell_init_files"], state_names)

# ── 6. Build the ALWAYS-two-cavity mechanics problem ──────────────────────
LV_ENDO_MARKER = biv_geo.markers["ENDO_LV"][0]
RV_ENDO_MARKER = biv_geo.markers["ENDO_RV"][0]

lv_v_target = dolfin.Constant(0.0)
rv_v_target = dolfin.Constant(0.0)

robin_bcs = list(mech_problem_base.bcs.robin)
cavity_bcs = pulse.BoundaryConditions(robin=robin_bcs, dirichlet=[], neumann=[])

problem = CavityConstrainedMechanicsProblem(
    biv_geo, mech_problem_base.material, cavity_bcs,
    cavity_specs={LV_ENDO_MARKER: lv_v_target, RV_ENDO_MARKER: rv_v_target},
    solver_parameters={"report": True, "absolute_tolerance": 1e-4, "relative_tolerance": 1e-4},
)

u0 = get_u_view(mech_problem_base)
dolfin.assign(problem.state.sub(0),
              dolfin.project(u0, dolfin.FunctionSpace(biv_geo.mechanics_mesh,
                                                        problem.state_space.ufl_element().sub_elements()[0])))

from simcardems.biv_cavity_cycle_controller import compute_cavity_volume
lvv_initial = compute_cavity_volume(biv_geo, u0, LV_ENDO_MARKER)
rvv_initial = compute_cavity_volume(biv_geo, u0, RV_ENDO_MARKER)
lv_v_target.assign(lvv_initial)
rv_v_target.assign(rvv_initial)
logger.info(f"Initial LV volume: {lvv_initial:.2f} mm^3, RV volume: {rvv_initial:.2f} mm^3")

nliter, nlconv = problem._raw_solve()
problem._update_active_stress_bookkeeping()
logger.info(f"Initial cavity-constrained solve: converged={nlconv} in {nliter} iters")

# ── 7. Circulation callback ────────────────────────────────────────────────
from pathlib import Path

outdir = Path(RESULTS_DIR)
from pathlib import Path

SAVE_FREQ_MS = 10.0  # matches your old config.save_freq convention

# PV loop CSV
pv_csv_path = Path(outdir) / "pv_loop.csv"
if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
    pv_file = open(pv_csv_path, "w")
    pv_file.write("t_ms,V_LV_mL,V_RV_mL,p_LV_mmHg,p_RV_mmHg\n")
    pv_file.flush()
else:
    pv_file = None

# Mesh deformation (u) — P1-projected for ParaView safety, plain XDMFFile.write()
# (write_checkpoint's P2 encoding is not ParaView-readable — see earlier debugging)
V_u_p1 = dolfin.VectorFunctionSpace(biv_geo.mechanics_mesh, "CG", 1)
u_p1 = dolfin.Function(V_u_p1)
u_p1.rename("u", "")
u_xdmf = dolfin.XDMFFile(str(Path(outdir) / "mechanics_u.xdmf"))

# Ta — already CG1, safe to write directly
active = problem.material.active
active.Ta_current.rename("Ta", "")
ta_xdmf = dolfin.XDMFFile(str(Path(outdir) / "mechanics_Ta.xdmf"))

_last_saved_t_ms = [-SAVE_FREQ_MS]  # list so the closure below can mutate it

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
pseudo_ecg.open_file(Path(outdir) / "pseudo_ecg.csv")

DT_ECG_MS = 5.0
_last_ecg_t_ms = [-DT_ECG_MS]

def p_BiV_func(V_LV, V_RV, t):
    """
    V_LV, V_RV in mL (circulation package convention), t in seconds.
    Sub-loops EP at EP_SUBSTEP_DT, then ONE mechanics solve + bookkeeping.
    """
    t_ms = t * 1000.0
    t_prev_ms = getattr(p_BiV_func, "_t_prev_ms", t_ms)
    n_substeps = max(1, round((t_ms - t_prev_ms) / EP_SUBSTEP_DT))
    sub_dt = (t_ms - t_prev_ms) / n_substeps if n_substeps > 0 else 0.0

    t_sub = t_prev_ms
    for _ in range(n_substeps):
        t_next = t_sub + sub_dt
        coupling.t = t_next
        coupling.solve_ep((t_sub, t_next))
        coupling.update_prev_ep()
        coupling.ep_to_coupling()
        t_sub = t_next

    coupling.coupling_to_mechanics()

    lv_v_target.assign(V_LV * 1e3)  # mL -> mm^3; VERIFY against your mesh units
    rv_v_target.assign(V_RV * 1e3)

    nliter, nlconv = problem._raw_solve()
    if not nlconv:
        logger.info(f"  [p_BiV_func] WARNING: mechanics did not converge at t={t_ms:.2f}ms, iters={nliter}")
    problem._update_active_stress_bookkeeping()

    coupling.interpolate(coupling.lmbda_mech, coupling.lmbda_ep)

    p_lv = problem.get_cavity_pressure(LV_ENDO_MARKER)
    p_rv = problem.get_cavity_pressure(RV_ENDO_MARKER)

    p_BiV_func._t_prev_ms = t_ms

    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0 and pv_file is not None:
        pv_file.write(f"{t_ms:.3f},{V_LV:.4f},{V_RV:.4f},"
                      f"{circulation.units.kPa_to_mmHg(p_lv):.4f},"
                      f"{circulation.units.kPa_to_mmHg(p_rv):.4f}\n")
        pv_file.flush()

    if t_ms - _last_saved_t_ms[0] >= SAVE_FREQ_MS - 1e-9:
        u_current = get_u_view(problem)
        u_p1.assign(dolfin.project(u_current, V_u_p1))
        u_xdmf.write(u_p1, t_ms)
        ta_xdmf.write(active.Ta_current, t_ms)
        _last_saved_t_ms[0] = t_ms

    if t_ms - _last_ecg_t_ms[0] >= DT_ECG_MS - 1e-9:
        coupling.assigners.assign_ep()
        v_fn = coupling.assigners.functions["ep"]["V"]
        pseudo_ecg.compute(v_fn, t_ms)
        _last_ecg_t_ms[0] = t_ms

    return circulation.units.kPa_to_mmHg(p_lv), circulation.units.kPa_to_mmHg(p_rv)


init_state = {"V_LV": lvv_initial / 1e3, "V_RV": rvv_initial / 1e3}  # mm^3 -> mL

circulation_model_3D = circulation.regazzoni2020.Regazzoni2020(
    add_units=False,
    p_BiV_func=p_BiV_func,
    verbose=True,
    comm=dolfin.MPI.comm_world,
    outdir=outdir,
    initial_state=init_state,
)

try:
    circulation_model_3D.solve(num_cycles=NUM_BEATS, initial_state=init_state, dt=CIRC_DT / 1000.0)
    # was: num_beats=NUM_BEATS
    circulation_model_3D.print_info()
except Exception as e:
    logger.info(f"Runner error: {e}")
finally:
    pseudo_ecg.close()
    if pv_file is not None:
        pv_file.close()
    logger.info("Done.")

logger.info(f"TOTAL SCRIPT WALLCLOCK TIME: {time.time() - t_script_start:.2f}s")