import os
import json
import dataclasses
import time

import dolfin
import pandas as pd

from simcardems.runner import Runner
from simcardems.time_stepper import TimeStepper
from simcardems.biv_cavity_cycle_controller import (
    Phase, compute_cavity_volume, compute_target_pressure, advance_phase, commit_step,
)
from simcardems import utils

from checkpoint_coord import save_function_by_coordinate, _field_stats, _leaf_stats_direct
from cavity_mechanics import get_u_view, _cheap_ta_now, solve_cavity_with_ta_ramp

logger = utils.getLogger(__name__)

def dt_targets_for_phase(phase):
    if phase == Phase.PRELOAD:
        return dict(dt=1.0, dt_mech=10.0)
    elif phase == Phase.EJECTION:
        return dict(dt=0.1, dt_mech=2.0)
    else:  # ISOVOL_CONTRACTION, ISOVOL_RELAXATION, FILLING
        return dict(dt=0.05, dt_mech=1.0)


def apply_phase_dt(config, phase, time_stepper=None, logger=None):
    targets = dt_targets_for_phase(phase)
    changed = (config.dt != targets["dt"]) or (config.dt_mech != targets["dt_mech"])
    if changed and logger is not None:
        logger.info(f"dt update (phase={phase}): dt {config.dt} -> {targets['dt']}, "
                    f"dt_mech {config.dt_mech} -> {targets['dt_mech']}")
    config.dt = targets["dt"]
    config.dt_mech = targets["dt_mech"]
    if time_stepper is not None:
        time_stepper.dt = TimeStepper.ms2ns(targets["dt"])
    return changed


class BiVCycleRunner(Runner):
    def set_cycle_controller(self, controller, mech_problem, outdir,
                              lv_pressure_const, rv_pressure_const,
                              state_names,
                              pseudo_ecg=None, dt_ecg=5.0,
                              warm_start_freq=5.0, t_restart=0.0):
        self._cycle_controller = controller
        self._mech_problem = mech_problem
        self._lv_pressure_const = lv_pressure_const
        self._rv_pressure_const = rv_pressure_const
        self._outdir = outdir
        self._state_names = state_names  # NEW: passed explicitly, was a module global before the split
        self._pseudo_ecg = pseudo_ecg
        self._dt_ecg = dt_ecg
        self._last_ecg_t = -dt_ecg
        self._warm_start_freq = warm_start_freq
        self._last_warm_start_t = t_restart
        self._last_solved_ta_max = None
        self.ta_trigger_threshold = 0.5

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

    def _solve_mechanics_now(self) -> bool:
        lv_state = self._cycle_controller.lv_state
        rv_state = self._cycle_controller.rv_state
        isovol_phases = (Phase.ISOVOL_CONTRACTION, Phase.ISOVOL_RELAXATION)
        in_isovol = (lv_state.phase in isovol_phases) or (rv_state.phase in isovol_phases)
        elapsed_trigger = self.coupling.dt_mechanics >= self._config.dt_mech

        if not in_isovol:
            return elapsed_trigger

        ta_now_max, ta_now_min = _cheap_ta_now(self._mech_problem)
        if self._last_solved_ta_max is None:
            ta_trigger = True
        else:
            ta_trigger = abs(ta_now_max - self._last_solved_ta_max) > self.ta_trigger_threshold

        triggered = elapsed_trigger or ta_trigger
        if triggered:
            logger.info(f"  [mechanics trigger] elapsed={elapsed_trigger}, ta_trigger={ta_trigger} "
                        f"(ta_now_max={ta_now_max:.4f}, ta_now_min={ta_now_min:.4f}, "
                        f"last_solved={self._last_solved_ta_max})")
        return triggered

    def _solve_mechanics(self):
        self.coupling.coupling_to_mechanics()
        t0 = time.time()

        lv_state = self._cycle_controller.lv_state
        rv_state = self._cycle_controller.rv_state
        isovol_phases = (Phase.ISOVOL_CONTRACTION, Phase.ISOVOL_RELAXATION)
        lv_isovol = lv_state.phase in isovol_phases
        rv_isovol = rv_state.phase in isovol_phases

        t_ms = TimeStepper.ns2ms(self.t)
        dt_ms = self._config.dt

        logger.info(f"  [dual-cavity] t={t_ms:.2f}ms  LV_phase={lv_state.phase} "
                    f"(isovol={lv_isovol})  RV_phase={rv_state.phase} (isovol={rv_isovol})")

        active_problem = self._cavity_manager.get_problem(lv_isovol, rv_isovol)
        logger.info(f"  [dual-cavity] using problem key={self._cavity_manager._current_key} "
                    f"(is_base_problem={active_problem is self._cavity_manager.base_problem})")

        if lv_isovol:
            target = lv_state.end_dia_vol if lv_state.phase == Phase.ISOVOL_CONTRACTION else lv_state.end_sys_vol
            self._cavity_manager.lv_v_target.assign(target)
            logger.info(f"  [dual-cavity] LV target_vol={target:.1f} mm^3 (Lagrange multiplier mode)")
        if rv_isovol:
            target = rv_state.end_dia_vol if rv_state.phase == Phase.ISOVOL_CONTRACTION else rv_state.end_sys_vol
            self._cavity_manager.rv_v_target.assign(target)
            logger.info(f"  [dual-cavity] RV target_vol={target:.1f} mm^3 (Lagrange multiplier mode)")

        if not lv_isovol:
            u_now = get_u_view(active_problem)
            v_lv_now = compute_cavity_volume(self._cycle_controller.geometry, u_now, self._cycle_controller.lv_marker)
            p_lv_target = compute_target_pressure(lv_state, v_lv_now, t_ms, dt_ms)
            self._lv_pressure_const.assign(p_lv_target)
            logger.info(f"  [dual-cavity] LV pressure-driven: v_lv_now={v_lv_now:.1f}, "
                        f"p_lv_target={p_lv_target:.4f} kPa (Neumann BC mode)")
        if not rv_isovol:
            u_now = get_u_view(active_problem)
            v_rv_now = compute_cavity_volume(self._cycle_controller.geometry, u_now, self._cycle_controller.rv_marker)
            p_rv_target = compute_target_pressure(rv_state, v_rv_now, t_ms, dt_ms)
            self._rv_pressure_const.assign(p_rv_target)
            logger.info(f"  [dual-cavity] RV pressure-driven: v_rv_now={v_rv_now:.1f}, "
                        f"p_rv_target={p_rv_target:.4f} kPa (Neumann BC mode)")

        if lv_isovol or rv_isovol:
            n_substeps, active_problem = solve_cavity_with_ta_ramp(
                self._cavity_manager, lv_isovol, rv_isovol, coupling=self.coupling)
            logger.info(f"  [dual-cavity] cavity-constrained solve done, {n_substeps} total Ta-ramp substeps")
        else:
            active_problem.solve()
            logger.info(f"  [dual-cavity] plain pressure-driven solve done")

        self._last_solved_ta_max, last_ta_min = _cheap_ta_now(self._mech_problem)
        logger.info(f"  [dual-cavity] post-solve Ta range: min={last_ta_min:.4f}, max={self._last_solved_ta_max:.4f}")
        u_final = get_u_view(active_problem)

        self.coupling.interpolate(self.coupling.lmbda_mech, self.coupling.lmbda_ep)

        v_lv_final = compute_cavity_volume(self._cycle_controller.geometry, u_final, self._cycle_controller.lv_marker)
        v_rv_final = compute_cavity_volume(self._cycle_controller.geometry, u_final, self._cycle_controller.rv_marker)
        p_lv_final = (active_problem.get_cavity_pressure(self._cycle_controller.lv_marker)
                      if lv_isovol else float(self._lv_pressure_const))
        p_rv_final = (active_problem.get_cavity_pressure(self._cycle_controller.rv_marker)
                      if rv_isovol else float(self._rv_pressure_const))

        logger.info(f"  [dual-cavity] FINAL this step: LV p={p_lv_final:.4f} kPa v={v_lv_final:.1f} mm^3  |  "
                    f"RV p={p_rv_final:.4f} kPa v={v_rv_final:.1f} mm^3")

        commit_step(lv_state, v_lv_final, p_lv_final)
        commit_step(rv_state, v_rv_final, p_rv_final)
        lv_state.end_dia_vol = v_lv_final
        rv_state.end_dia_vol = v_rv_final

        lv_phase_before = lv_state.phase
        rv_phase_before = rv_state.phase
        advance_phase(lv_state, t_ms, dt_ms)
        advance_phase(rv_state, t_ms, dt_ms)

        if lv_state.phase != lv_phase_before:
            logger.info(f"  [dual-cavity] *** LV PHASE TRANSITION: {lv_phase_before} -> {lv_state.phase} *** "
                        f"(pressure_n={lv_state.pressure_n:.4f}, wdk_pressure_n={lv_state.wdk_pressure_n:.4f})")
        if rv_state.phase != rv_phase_before:
            logger.info(f"  [dual-cavity] *** RV PHASE TRANSITION: {rv_phase_before} -> {rv_state.phase} *** "
                        f"(pressure_n={rv_state.pressure_n:.4f}, wdk_pressure_n={rv_state.wdk_pressure_n:.4f})")

        self.coupling.update_prev_mechanics()
        self.coupling.mechanics_to_coupling()
        self.coupling.coupling_to_ep()

        apply_phase_dt(self._config, lv_state.phase, time_stepper=self._time_stepper, logger=logger)

        t1 = time.time()
        logger.debug(f"  Mechanics solve time: {t1 - t0:.2f}s")
        logger.debug(
            f"  → t={t_ms:.1f} ms"
            f"  LV phase={lv_state.phase}  LVP={lv_state.pressure_n:.3f} kPa  LVV={lv_state.volume_n / 1000:.2f} mL"
            f"  RV phase={rv_state.phase}  RVP={rv_state.pressure_n:.3f} kPa  RVV={rv_state.volume_n / 1000:.2f} mL"
        )

        t_ms_now = TimeStepper.ns2ms(self.t)
        if (t_ms_now - self._last_warm_start_t) >= (self._warm_start_freq - 1e-10):
            checkpoint_name = f"warm_start_{int(t_ms_now):04d}ms"
            checkpoint_base = os.path.join(self._outdir, checkpoint_name)

            save_function_by_coordinate(self.coupling.ep_solver.vs, checkpoint_base + "_vs")
            save_function_by_coordinate(self.coupling.mech_solver.state, checkpoint_base + "_mechstate")
            save_function_by_coordinate(self.coupling.lmbda_mech, checkpoint_base + "_lmbda")
            save_function_by_coordinate(self.coupling.Zetas_mech, checkpoint_base + "_zetas")
            save_function_by_coordinate(self.coupling.Zetaw_mech, checkpoint_base + "_zetaw")
            save_function_by_coordinate(self.coupling.mech_solver.material.active.Ta_current, checkpoint_base + "_ta")

            self.coupling.assigners.assign_ep()
            checksums = {
                "u": _field_stats(get_u_view(self._mech_problem)),
                "lmbda_mech": _field_stats(self.coupling.lmbda_mech),
                "Zetas_mech": _field_stats(self.coupling.Zetas_mech),
                "Zetaw_mech": _field_stats(self.coupling.Zetaw_mech),
                "Ta_current": _field_stats(self.coupling.mech_solver.material.active.Ta_current),
            }
            for fname in ["cai", "XS", "XW", "CaTrpn", "TmB", "Cd", "v"]:
                idx = self._state_names.index(fname)
                checksums[f"vs_{fname}"] = _leaf_stats_direct(self.coupling.ep_solver.vs, idx)

            if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
                with open(checkpoint_base + ".json", "w") as f:
                    json.dump({
                        "t_ms": t_ms_now,
                        "lv": {k: v for k, v in dataclasses.asdict(lv_state).items() if not isinstance(v, dict)},
                        "rv": {k: v for k, v in dataclasses.asdict(rv_state).items() if not isinstance(v, dict)},
                        "checksums": checksums,
                    }, f, indent=2)
            self._last_warm_start_t = t_ms_now

        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0 and self._pv_file is not None:
            self._pv_file.write(
                f"{t_ms_now:.3f},{lv_state.pressure_n:.6f},{lv_state.volume_n:.6f},"
                f"{rv_state.pressure_n:.6f},{rv_state.volume_n:.6f},"
                f"{lv_state.phase},{rv_state.phase}\n"
            )
            self._pv_file.flush()

        if self._pseudo_ecg is not None and t_ms_now - self._last_ecg_t >= self._dt_ecg - 1e-10:
            self.coupling.assigners.assign_ep()
            v_fn = self.coupling.assigners.functions["ep"]["V"]
            self._pseudo_ecg.compute(v_fn, t_ms_now)
            self._last_ecg_t = t_ms_now

    # def _post_ep(self):
    #     super()._post_ep()
    #     t_ms = TimeStepper.ns2ms(self.t)
    #
    #     self.coupling.assigners.assign_ep()
    #     cai_fn = self.coupling.assigners.functions["ep"]["Ca"]
    #     xs_fn = self.coupling.assigners.functions["ep"]["XS"]
    #     cd_fn = self.coupling.assigners.functions["ep"]["Cd"]
    #     catrpn_fn = self.coupling.assigners.functions["ep"]["CaTrpn"]
    #     tmb_fn = self.coupling.assigners.functions["ep"]["TmB"]
    #
    #     cai_now = dolfin.MPI.max(dolfin.MPI.comm_world, float(cai_fn.vector().get_local().max()))
    #     xs_now = dolfin.MPI.max(dolfin.MPI.comm_world, float(xs_fn.vector().get_local().max()))
    #     cd_now = dolfin.MPI.max(dolfin.MPI.comm_world, float(cd_fn.vector().get_local().max()))
    #     catrpn_now = dolfin.MPI.max(dolfin.MPI.comm_world, float(catrpn_fn.vector().get_local().max()))
    #     tmb_max = dolfin.MPI.max(dolfin.MPI.comm_world, float(tmb_fn.vector().get_local().max()))
    #     tmb_min = dolfin.MPI.min(dolfin.MPI.comm_world, float(tmb_fn.vector().get_local().min()))
    #
    #     active_problem = self._cavity_manager._current_problem
    #     active = active_problem.material.active
    #     u = get_u_view(active_problem)
    #     F = dolfin.grad(u) + dolfin.Identity(3)
    #     f = F * active.f0
    #     lmbda_expr = dolfin.sqrt(f ** 2)
    #     lmbda_scratch = dolfin.Function(active.Ta_current.function_space())
    #     active._projector.project(lmbda_scratch, lmbda_expr)
    #     lmbda_fresh_max = dolfin.MPI.max(dolfin.MPI.comm_world, float(lmbda_scratch.vector().max()))
    #
    #     # if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
    #     #     logger.info(f"  [live EP debug] t={t_ms:.3f}ms  Ca_max={cai_now:.6e}  XS_max={xs_now:.6e}  "
    #     #                 f"Cd_max={cd_now:.6e}  CaTrpn_max={catrpn_now:.6e}  "
    #     #                 f"TmB_max={tmb_max:.6e} TmB_min={tmb_min:.6e}  lambda_max={lmbda_fresh_max:.6f}")

    def close_files(self):
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0 and self._pv_file is not None:
            self._pv_file.close()
        if self._pseudo_ecg is not None:
            self._pseudo_ecg.close()