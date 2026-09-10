import os
import json
import dataclasses
import time

import dolfin
import pandas as pd
import numpy as np

from simcardems.runner import Runner
from simcardems.time_stepper import TimeStepper
from simcardems.biv_cavity_cycle_controller import (
    Phase, compute_cavity_volume, compute_target_pressure, advance_phase, commit_step,
)
from simcardems import utils

from checkpoint_coord import save_function_by_coordinate, _field_stats, _leaf_stats_direct
from cavity_mechanics import get_u_view, _cheap_ta_now, solve_cavity_with_ta_ramp

logger = utils.getLogger(__name__)


def dt_targets_for_phase(phase, t_ms):
    if (phase == Phase.PRELOAD) & (t_ms < 10):
        return dict(dt=1.0, dt_mech=1.0)
    elif (phase == Phase.PRELOAD) & (t_ms >= 10):
        return dict(dt=1.0, dt_mech=5.0)
    elif phase == Phase.EJECTION:
        return dict(dt=0.1, dt_mech=2.0)
    else:  # ISOVOL_CONTRACTION, ISOVOL_RELAXATION, FILLING
        return dict(dt=0.05, dt_mech=0.1)


def apply_phase_dt(config, phase, t_ms, time_stepper=None, logger=None):
    targets = dt_targets_for_phase(phase, t_ms)
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
        logger.info("  [debug] entering set_cycle_controller")
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
        V_mech_lambda = self.coupling.mech_solver.material.active.lmbda.function_space()
        self._lambda_diff_fn = dolfin.Function(V_mech_lambda, name="lambda_diff")
        self.collector.register("mechanics", "lambda_diff", self._lambda_diff_fn)
        shared_V = self.coupling.mech_solver.material.active.Ta_current.function_space()
        self._detF_export_fn = dolfin.Function(shared_V, name="detF")
        self.collector.register("mechanics", "detF", self._detF_export_fn)

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

    def _export_strain_stress(self, active_problem):
        from cavity_mechanics import get_u_view, get_p_view

        material = active_problem.material
        active = material.active

        u = get_u_view(active_problem)
        p = get_p_view(active_problem)

        F = dolfin.variable(dolfin.grad(u) + dolfin.Identity(3))
        C = F.T * F
        E = 0.5 * (C - dolfin.Identity(3))
        P_passive = material.FirstPiolaStress(F, p)

        f0 = active.f0
        f = F * f0
        valve_mask = getattr(self._cycle_controller.geometry, "valve_mask", None)
        if valve_mask is not None:
            myocardium = 1.0 - valve_mask
            P_active = myocardium * active.Ta_current * dolfin.outer(f, f0)
        else:
            P_active = active.Ta_current * dolfin.outer(f, f0)
        P_total = P_passive + P_active

        idx = 0
        for i in range(3):
            for j in range(3):
                self.coupling._E_components[idx].assign(
                    dolfin.project(E[i, j], self.coupling._E_components[idx].function_space()))
                self.coupling._P_components[idx].assign(
                    dolfin.project(P_total[i, j], self.coupling._P_components[idx].function_space()))
                idx += 1

    def _export_lambda_diff(self, active_problem):
        active = active_problem.material.active
        V_mech = active.lmbda.function_space()
        V_ep = self.coupling.lmbda_ep.function_space()

        mech_vals = active.lmbda.vector().get_local()
        ep_vals = self.coupling.lmbda_ep.vector().get_local()

        mech_coords = V_mech.tabulate_dof_coordinates()
        ep_coords = V_ep.tabulate_dof_coordinates()

        from scipy.spatial import cKDTree
        tree = cKDTree(ep_coords)
        _, nn_idx = tree.query(mech_coords)
        diff = mech_vals - ep_vals[nn_idx]

        self._lambda_diff_fn.vector().set_local(diff)
        self._lambda_diff_fn.vector().apply("insert")

        local_max_abs = float(np.abs(diff).max()) if len(diff) else 0.0
        global_max_abs = dolfin.MPI.max(dolfin.MPI.comm_world, local_max_abs)
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            logger.info(f"  [lambda diff] max|mech-ep| across whole mesh = {global_max_abs:.6f}")

    def _check_detF_full_mesh(self, active_problem):
        from cavity_mechanics import get_u_view, get_p_view

        u = get_u_view(active_problem)
        p = get_p_view(active_problem)

        F = dolfin.grad(u) + dolfin.Identity(3)
        J = dolfin.det(F)

        self._detF_export_fn.assign(dolfin.project(J, self._detF_export_fn.function_space()))
        J_vals = self._detF_export_fn.vector().get_local()

        p_vals = p.vector().get_local()

        mesh = (active_problem.geometry.mechanics_mesh
                if hasattr(active_problem.geometry, "mechanics_mesh")
                else active_problem.geometry.mesh)
        comm = mesh.mpi_comm()

        local_min_J = float(J_vals.min()) if len(J_vals) else float('inf')
        local_max_J = float(J_vals.max()) if len(J_vals) else float('-inf')
        global_min_J = dolfin.MPI.min(comm, local_min_J)
        global_max_J = dolfin.MPI.max(comm, local_max_J)

        n_below_1_local = int(np.sum(J_vals < 1.0))
        n_negative_local = int(np.sum(J_vals < 0.0))
        n_below_1_total = dolfin.MPI.sum(comm, n_below_1_local)
        n_negative_total = dolfin.MPI.sum(comm, n_negative_local)

        rank = dolfin.MPI.rank(comm)
        if rank == 0:
            logger.info(f"  [detF full-mesh check] GLOBAL: J min={global_min_J:.6f}, max={global_max_J:.6f}, "
                        f"total n<1.0={int(n_below_1_total)}, total n<0={int(n_negative_total)}")

        if n_negative_local > 0:
            bad_idx = np.where(J_vals < 0.0)[0]
            cell_centers = np.array([dolfin.Cell(mesh, int(i)).midpoint().array() for i in bad_idx])
            logger.info(f"  [detF full-mesh check] rank{rank}: {len(bad_idx)} NEGATIVE detF cells at:\n{cell_centers}")

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

    def _trace_cell_lambda(self, active_problem, label=""):
        active = active_problem.material.active
        V_mech = active.lmbda.function_space()
        V_ep = self.coupling.lmbda_ep.function_space()

        mesh = V_mech.mesh()
        target_point = np.array([90.0, 55.0, 65.0])  # replace with the real centroid of your selected element

        v2d_mech = dolfin.vertex_to_dof_map(V_mech)
        v2d_ep = dolfin.vertex_to_dof_map(V_ep)
        coords = mesh.coordinates()

        if len(coords) == 0:
            return  # this rank owns no local vertices near anywhere relevant

        dists = np.linalg.norm(coords - target_point, axis=1)
        nearest_vid = int(np.argmin(dists))
        nearest_dist = dists[nearest_vid]

        comm = mesh.mpi_comm()
        global_min_dist = dolfin.MPI.min(comm, float(nearest_dist))

        # Only the rank that actually owns the closest vertex reports a value
        if abs(nearest_dist - global_min_dist) < 1e-9:
            mech_vals_local = active.lmbda.vector().get_local()
            ep_vals_local = self.coupling.lmbda_ep.vector().get_local()
            ta_vals_local = active.Ta_current.vector().get_local()

            dof_mech = v2d_mech[nearest_vid]
            dof_ep = v2d_ep[nearest_vid]

            m_val = mech_vals_local[dof_mech]
            e_val = ep_vals_local[dof_ep]
            ta_val = ta_vals_local[dof_mech]

            logger.info(f"  [vertex trace, {label}] at {coords[nearest_vid]}: "
                        f"mech_lambda={m_val:.6f}, ep_lambda={e_val:.6f}, Ta_current={ta_val:.6f}")

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

        # logger.info(
        #     f"  [solver check] right before solve: linear_solver={active_problem.solver.parameters['linear_solver']}, "
        #     f"preconditioner={active_problem.solver.parameters['preconditioner']}")

        if lv_isovol or rv_isovol:
            n_substeps, active_problem = solve_cavity_with_ta_ramp(self._cavity_manager, lv_isovol, rv_isovol,
                                                                   coupling=self.coupling)
            logger.info(f"  [dual-cavity] cavity-constrained solve done, {n_substeps} total Ta-ramp substeps")
        else:
            active_problem.solve()
            logger.info(f"  [dual-cavity] plain pressure-driven solve done")

        # self._check_detF_full_mesh(active_problem)
        # self._export_strain_stress(active_problem)

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

        apply_phase_dt(self._config, lv_state.phase, t_ms, time_stepper=self._time_stepper, logger=logger)

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

    def close_files(self):
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0 and self._pv_file is not None:
            self._pv_file.close()
        if self._pseudo_ecg is not None:
            self._pseudo_ecg.close()