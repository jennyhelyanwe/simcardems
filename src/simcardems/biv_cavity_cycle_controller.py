"""
Biventricular cardiac cycle pressure controller for simcardems/pulse.

Architecture:
  - CavityState: per-cavity (LV, RV) phase machine + Windkessel ODE state,
    independent — each cavity tracks its own phase, so LV and RV can be
    in different phases simultaneously (matches Alya's icavi-indexed
    cavities(:) array, generalized to allow phase desync).
  - compute_cavity_volume: from the previous turn, Nanson's-formula-correct
    divergence-theorem volume from the reference-config plane normal.
  - BiVCycleController: owns both CavityStates, computes both target
    pressures each macro step, then issues ONE pulse.iterate.iterate()
    call moving (LV pressure, RV pressure) together as a tuple control,
    matching iterate()'s documented (LVP, RVP) support.

NOT YET VERIFIED:
  - Exact field names/types iterate() expects for a tuple control of two
    dolfin.Constants (the docstring says "tuple if target is (LVP, RVP)"
    but the actual zip/enlist mechanics in Iterator weren't traced in
    detail against two independent dolfin.Constant objects specifically —
    worth a small isolated test before trusting this on a full run).
  - Plane-normal/plane-points definition for your BiV mesh (LV/RV valve
    planes) — placeholder API below, needs your actual reference node
    coordinates per cavity.
"""

import dataclasses
import typing

import dolfin

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

import pulse
from pulse import kinematics
from . import utils


logger = utils.getLogger(__name__)

# --- Cavity volume (from previous turn, included for completeness) -----

def compute_cavity_volume(geometry, u, endo_marker):
    F = dolfin.variable(kinematics.DeformationGradient(u))
    N = geometry.facet_normal
    ds = geometry.ds

    n_weighted = ufl.cofac(F) * N
    x = dolfin.SpatialCoordinate(geometry.mesh) + u

    integrand = -dolfin.dot(x, n_weighted) / 3.0
    volume = dolfin.assemble(integrand * ds(endo_marker))
    return volume

def compute_cavity_volume_open(geometry, u, marker, plane_normal):
    F = dolfin.variable(kinematics.DeformationGradient(u))
    N = geometry.facet_normal
    ds = geometry.ds

    n_current_weighted = ufl.cofac(F) * N
    x = dolfin.SpatialCoordinate(geometry.mesh) + u
    n_hat = dolfin.Constant(plane_normal)

    integrand = -dolfin.dot(x, n_hat) * dolfin.dot(n_current_weighted, n_hat)
    return dolfin.assemble(integrand * ds(marker))


# --- Phase enum, matching Alya's integer phase codes exactly -----------

class Phase:
    PRELOAD = 0
    ISOVOL_CONTRACTION = 1
    EJECTION = 2
    ISOVOL_RELAXATION = 3
    FILLING = 4


@dataclasses.dataclass
class WindkesselParams:
    p_init: float       # ini_wdk
    compliance: float   # c_wdk
    resistance: float   # r_wdk
    evolve: bool = True  # evolve_aop


@dataclasses.dataclass
class CycleParams:
    """Per-cavity parameters, mirroring cycles(id) % ... fields."""
    t_zero: float            # tzero
    t_prestress: float       # tpstr
    preload_pressure: float  # pzero
    prestress_pressure: float  # pstr0
    t_end_diastole: float    # tendd
    p_end_diastole: float    # pendd
    gain_contraction: typing.Tuple[float, float]  # gains_contraction (err, derr)
    gain_relaxation: typing.Tuple[float, float]   # gains_relaxation (err, derr)
    p_fill: float             # ppost
    period: float             # perio
    windkessel: WindkesselParams
    filling_gain: bool = False  # cycles(id) % filling_gain


@dataclasses.dataclass
class CavityState:
    """
    Per-cavity persistent state across the time loop, mirroring Alya's
    cavities(icavi) struct. One instance per cavity (LV, RV) — phases
    evolve independently.
    """
    name: str
    params: CycleParams

    phase: int = Phase.PRELOAD
    n_beats: int = 0
    phase_counter: int = 0
    last_phase_change: float = 0.0

    volume_n: float = 0.0
    volume_n_minus_1: float = 0.0
    dvol_n: float = 0.0

    pressure_n: float = 0.0
    prestress_n: float = 0.0
    pressure_n_minus_1: float = 0.0

    wdk_pressure_n: float = 0.0

    ini_vol: float = 0.0
    end_preload_vol: float = 0.0
    end_dia_vol: float = 0.0
    end_sys_vol: float = 0.0

    initialized: bool = False

    def initialize(self, volume_0: float):
        """Call once, at the very first time step (ittim==1 equivalent)."""
        self.wdk_pressure_n = self.params.windkessel.p_init
        self.ini_vol = volume_0
        self.volume_n = volume_0
        self.volume_n_minus_1 = volume_0
        self.initialized = True


# --- Single-cavity pressure update (the ITASK_ENDINN logic) ------------

def _update_windkessel(state: CavityState, dvol: float, dt: float) -> float:
    """One Windkessel ODE step. Returns new wdk pressure (ITER_K value)."""
    wdk = state.params.windkessel
    if not wdk.evolve:
        if state.phase != Phase.EJECTION:
            return wdk.p_init
    if state.phase == Phase.EJECTION:
        return state.wdk_pressure_n - (dt / wdk.compliance) * (
            dvol + state.wdk_pressure_n / wdk.resistance
        )
    return state.wdk_pressure_n


def compute_target_pressure(
    state: CavityState,
    volume_iter_k: float,
    t: float,
    dt: float,
) -> float:
    """
    Compute the new target cavity pressure for this macro step, given the
    just-converged-or-trial volume at the current time. Mirrors the
    case(cavities(icavi)%phase) block in sld_cardiac_phases.

    NOTE: Alya recomputes this every inner Newton iteration (ITASK_ENDINN);
    here it's called once per macro step using the volume from the
    previous converged step (since with iterate()-based control we don't
    have an equivalent inner-iteration hook — see architecture note in
    the module docstring). This is a real simplification relative to
    Alya's tighter coupling, worth validating against a known PV-loop
    benchmark once running, not just trusting structurally.
    """
    p = state.params
    dvol = volume_iter_k - state.volume_n
    ddvol = dvol / dt

    wdk_pres = _update_windkessel(state, dvol if state.phase == Phase.EJECTION else 0.0, dt)

    if state.phase == Phase.PRELOAD:
        if t <= p.t_zero:
            pressure = p.preload_pressure * (t / p.t_zero)
            if p.prestress_pressure > 0.1:
                if t < p.t_prestress:
                    state.prestress_n = p.prestress_pressure * (t / p.t_prestress)
                elif t < p.t_zero:
                    c_vol = state.volume_n / max(state.prestress_n, 1e-12)
                    c_vel = 1.0
                    dvol_aux = state.volume_n - state.ini_vol
                    new_prestress = state.prestress_n + dvol_aux / c_vol + ddvol / c_vel
                    if new_prestress < 1.0:
                        raise RuntimeError(
                            f"{state.name}: negative pre-stress encountered"
                        )
                    state.prestress_n = new_prestress
            state.end_preload_vol = volume_iter_k
        elif p.t_end_diastole > p.t_zero and t > p.t_zero:
            pressure = (
                (p.p_end_diastole - p.preload_pressure)
                * (t - state.n_beats * p.period - p.t_zero)
                / (p.t_end_diastole - p.t_zero)
                + p.preload_pressure
            )
        else:
            pressure = state.pressure_n
        state.end_dia_vol = volume_iter_k

    elif state.phase == Phase.ISOVOL_CONTRACTION:
        logger.info("Unused IVC code")

    elif state.phase == Phase.EJECTION:
        pressure = wdk_pres
        state.end_sys_vol = state.volume_n


    elif state.phase == Phase.ISOVOL_RELAXATION:
        logger.info("Unused IVR code")

    elif state.phase == Phase.FILLING:
        if p.filling_gain:
            dvol_aux = state.volume_n - state.end_preload_vol
            pressure = (
                state.pressure_n
                - p.gain_relaxation[1] * ddvol
                - p.gain_relaxation[0] * dvol_aux
            )
            if state.volume_n > state.end_preload_vol:
                pressure = state.pressure_n - p.gain_relaxation[0] * dvol_aux
        else:
            fill_rate = (p.preload_pressure - p.p_fill) / max(
                abs(state.ini_vol - state.end_sys_vol), 1e-6
            )
            fill_rate = max(fill_rate, -500.0)
            pressure = state.pressure_n + fill_rate * dvol
            pressure = max(pressure, p.preload_pressure)
    else:
        raise ValueError(f"Unknown phase {state.phase}")

    state.wdk_pressure_n = wdk_pres
    return pressure


def advance_phase(state: CavityState, t: float, dt: float):
    """
    Phase transition logic, mirroring sld_cardiac_phase_transition.
    Call AFTER compute_target_pressure and AFTER the solver has converged
    for this step (i.e. once volume_n/pressure_n reflect the converged state).
    """
    p = state.params
    delta_phase_change = t - state.last_phase_change

    if state.phase == Phase.PRELOAD and t >= state.n_beats * p.period + p.t_end_diastole and t >= p.t_zero:
        state.phase = Phase.ISOVOL_CONTRACTION
        state.last_phase_change = t

    elif state.phase == Phase.ISOVOL_CONTRACTION and state.pressure_n > state.wdk_pressure_n:
        state.phase = Phase.EJECTION
        state.last_phase_change = t

    elif (
        state.phase == Phase.EJECTION
        and state.dvol_n > 1e-5
        and delta_phase_change > 0.05
        and state.volume_n < 0.99 * state.end_dia_vol
    ):
        state.phase = Phase.ISOVOL_RELAXATION
        state.last_phase_change = t

    elif state.phase == Phase.ISOVOL_RELAXATION and state.pressure_n < p.p_fill:
        state.phase = Phase.FILLING
        state.phase_counter = 0
        state.last_phase_change = t

    elif state.phase == Phase.FILLING:
        if p.period > 0.0 and t >= (state.n_beats + 1) * p.period + p.t_zero:
            state.phase = Phase.PRELOAD
            state.last_phase_change = t
            state.n_beats += 1
        elif delta_phase_change > 0.05:
            if state.dvol_n < -1e-6:
                state.phase_counter += 1
            else:
                state.phase_counter = 0
            if state.phase_counter > 5:
                state.last_phase_change = t
                state.n_beats += 1
                state.phase_counter = 0


def commit_step(state: CavityState, new_volume: float, new_pressure: float):
    """End-of-macro-step bookkeeping, mirroring ITASK_ENDITE."""
    state.dvol_n = new_volume - state.volume_n
    state.volume_n_minus_1 = state.volume_n
    state.pressure_n_minus_1 = state.pressure_n
    state.volume_n = new_volume
    state.pressure_n = new_pressure


# --- BiV driver: independent phases, one combined iterate() call -------

class BiVCycleController:
    def __init__(
        self,
        lv_state: CavityState,
        rv_state: CavityState,
        lv_pressure_constant: dolfin.Constant,
        rv_pressure_constant: dolfin.Constant,
        geometry,
        lv_marker: int,
        rv_marker: int,
    ):
        self.lv_state = lv_state
        self.rv_state = rv_state
        self.lv_pressure_constant = lv_pressure_constant
        self.rv_pressure_constant = rv_pressure_constant
        self.geometry = geometry
        self.lv_marker = lv_marker
        self.rv_marker = rv_marker

    def initialize(self, u0: dolfin.Function):
        v_lv = compute_cavity_volume(self.geometry, u0, self.lv_marker)
        v_rv = compute_cavity_volume(self.geometry, u0, self.rv_marker)
        self.lv_state.initialize(v_lv)
        self.rv_state.initialize(v_rv)

    # def step(self, problem: pulse.MechanicsProblem, t: float, dt: float):
    #     """
    #     One macro time step:
    #       1. compute each cavity's current volume from the LAST converged state
    #       2. compute each cavity's target pressure independently (different phases OK)
    #       3. one combined iterate() call moving (LV, RV) pressure together
    #       4. recompute volumes from the new converged state, commit, advance phases
    #     """
    #     isovol_phases = (Phase.ISOVOL_CONTRACTION, Phase.ISOVOL_RELAXATION)
    #     if self.lv_state.phase in isovol_phases or self.rv_state.phase in isovol_phases:
    #         # Pressure is already converged externally (converge_isovolumic_pressure
    #         # in run_coarse.py, called BEFORE this step()). Do NOT recompute a
    #         # target or re-solve here. Just commit + check transitions.
    #         u_new, _ = problem.state.split(deepcopy=True)
    #         v_lv_new = compute_cavity_volume(self.geometry, u_new, self.lv_marker)
    #         v_rv_new = compute_cavity_volume(self.geometry, u_new, self.rv_marker)
    #         target_lv = float(self.lv_pressure_constant)
    #         target_rv = float(self.rv_pressure_constant)
    #         commit_step(self.lv_state, v_lv_new, target_lv)
    #         commit_step(self.rv_state, v_rv_new, target_rv)
    #         advance_phase(self.lv_state, t, dt)
    #         advance_phase(self.rv_state, t, dt)
    #         return
    #
    #     u, _ = problem.state.split(deepcopy=True)
    #     v_lv_now = compute_cavity_volume(self.geometry, u, self.lv_marker)
    #     v_rv_now = compute_cavity_volume(self.geometry, u, self.rv_marker)
    #
    #     target_lv = compute_target_pressure(self.lv_state, v_lv_now, t, dt)
    #     target_rv = compute_target_pressure(self.rv_state, v_rv_now, t, dt)
    #     # target_lv = 0.0
    #     # target_rv = 0.0
    #     # if self.lv_state.phase == Phase.PRELOAD and self.rv_state.phase == Phase.PRELOAD:
    #         # During preload assign directly — pressure ramps linearly and
    #         # is small, no need for pulse.iterate's cautious stepping
    #         # self.lv_pressure_constant.assign(target_lv)
    #         # self.rv_pressure_constant.assign(target_rv)
    #         # logger.debug(f"[PRELOAD] target_lv: {target_lv!r} kPa, target_rv: {target_rv!r} kPa")
    #         # problem.solve()
    #     if self.lv_state.phase == Phase.PRELOAD and self.rv_state.phase == Phase.PRELOAD:
    #         # Use pulse.iterate instead of direct assignment
    #         pulse.iterate.iterate(
    #             problem,
    #             control=(self.lv_pressure_constant, self.rv_pressure_constant),
    #             target=(target_lv, target_rv),
    #         )
    #     else:
    #         # self.lv_pressure_constant.assign(0.0)
    #         # self.rv_pressure_constant.assign(0.0)
    #         try:
    #             lv_unchanged = abs(target_lv - float(self.lv_pressure_constant)) < 1e-10
    #             rv_unchanged = abs(target_rv - float(self.rv_pressure_constant)) < 1e-10
    #
    #             if lv_unchanged and rv_unchanged:
    #                 print('Both LVP and RVP are unchanged')
    #                 problem.solve()
    #             elif lv_unchanged:
    #                 print('LVP is unchanged.')
    #                 self.lv_pressure_constant.assign(target_lv)
    #                 pulse.iterate.iterate(
    #                     problem,
    #                     control=self.rv_pressure_constant,
    #                     target=target_rv,
    #                 )
    #             elif rv_unchanged:
    #                 print('RVP is unchanged.')
    #                 self.rv_pressure_constant.assign(target_rv)
    #                 pulse.iterate.iterate(
    #                     problem,
    #                     control=self.lv_pressure_constant,
    #                     target=target_lv,
    #                 )
    #             else:
    #                 pulse.iterate.iterate(
    #                     problem,
    #                     control=(self.lv_pressure_constant, self.rv_pressure_constant),
    #                     target=(target_lv, target_rv),
    #                 )
    #         except ZeroDivisionError:
    #             import traceback
    #             traceback.print_exc()
    #             raise
    #
    #     # ── NEW: one correction pass, IVC/IVR only ──────────────────────
    #     isovol_phases = (Phase.ISOVOL_CONTRACTION, Phase.ISOVOL_RELAXATION)
    #     if self.lv_state.phase in isovol_phases or self.rv_state.phase in isovol_phases:
    #         u_trial, _ = problem.state.split(deepcopy=True)
    #         v_lv_trial = compute_cavity_volume(self.geometry, u_trial, self.lv_marker)
    #         v_rv_trial = compute_cavity_volume(self.geometry, u_trial, self.rv_marker)
    #
    #         # Recompute target pressure using the volume the trial solve
    #         # actually produced, rather than the volume from last step —
    #         # this is the "solve, correct, re-solve" pass Alya does via
    #         # ITER_K inside its inner Newton loop.
    #         corrected_lv = compute_target_pressure(self.lv_state, v_lv_trial, t, dt)
    #         corrected_rv = compute_target_pressure(self.rv_state, v_rv_trial, t, dt)
    #
    #         pulse.iterate.iterate(
    #             problem,
    #             control=(self.lv_pressure_constant, self.rv_pressure_constant),
    #             target=(corrected_lv, corrected_rv),
    #         )
    #         target_lv, target_rv = corrected_lv, corrected_rv
    #
    #     try:
    #         u_new, _ = problem.state.split(deepcopy=True)
    #         v_lv_new = compute_cavity_volume(self.geometry, u_new, self.lv_marker)
    #         v_rv_new = compute_cavity_volume(self.geometry, u_new, self.rv_marker)
    #         commit_step(self.lv_state, v_lv_new, target_lv)
    #         commit_step(self.rv_state, v_rv_new, target_rv)
    #         advance_phase(self.lv_state, t, dt)
    #         advance_phase(self.rv_state, t, dt)
    #     except ZeroDivisionError as e:
    #         import traceback
    #         traceback.print_exc()
    #         raise