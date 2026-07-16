import dolfin
import pulse

from . import utils


logger = utils.getLogger(__name__)


class MechanicsNewtonSolver(dolfin.NewtonSolver):
    def __init__(
        self,
        problem: pulse.NonlinearProblem,
        state: dolfin.Function,
        update_cb=None,
        parameters=None,
    ):
        logger.debug(f"Initialize NewtonSolver with parameters: {parameters!r}")
        dolfin.PETScOptions.clear()
        self._problem = problem
        self._state = state
        self._update_cb = update_cb

        # Initializing Newton solver (parent class)
        # self.petsc_solver = dolfin.PETScKrylovSolver()
        self.petsc_solver = dolfin.PETScKrylovSolver("gmres", "hypre_amg")
        super().__init__(
            self._state.function_space().mesh().mpi_comm(),
            self.petsc_solver,
            dolfin.PETScFactory.instance(),
        )
        self._handle_parameters(parameters)

    def _handle_parameters(self, parameters):
        # Setting default parameters
        params = MechanicsNewtonSolver_ODE.default_solver_parameters()
        params.update(parameters)

        for k, v in params.items():
            if self.parameters.has_parameter(k):
                self.parameters[k] = v
            if self.parameters.has_parameter_set(k):
                for subk, subv in params[k].items():
                    self.parameters[k][subk] = subv
        petsc = params.pop("petsc")
        for k, v in petsc.items():
            if v is not None:
                dolfin.PETScOptions.set(k, v)
        self.newton_verbose = params.pop("newton_verbose", False)
        self.ksp_verbose = params.pop("ksp_verbose", False)
        self.debug = params.pop("debug", False)
        if self.newton_verbose:
            dolfin.set_log_level(dolfin.LogLevel.INFO)
            self.parameters["report"] = True
        if self.ksp_verbose:
            self.parameters["lu_solver"]["report"] = True
            self.parameters["lu_solver"]["verbose"] = True
            self.parameters["krylov_solver"]["monitor_convergence"] = True
            dolfin.PETScOptions.set("ksp_monitor_true_residual")
        self.parameters["linear_solver"] = "gmres"
        self.parameters["preconditioner"] = "hypre_amg"
        self.linear_solver().set_from_options()
        self._residual_index = 0
        self._residuals = []

    def register_datacollector(self, datacollector):
        self._datacollector = datacollector

    # @staticmethod
    # def default_solver_parameters():
    #     return {
    #         "petsc": {
    #             "ksp_type": "preonly",
    #             # "ksp_type": "gmres",
    #             "pc_type": "lu",
    #             "pc_factor_mat_solver_type": "mumps",
    #             "mat_mumps_icntl_33": 0,
    #         },
    #         "newton_verbose": False,
    #         "ksp_verbose": False,
    #         "debug": False,
    #         "linear_solver": "mumps",
    #         # "linear_solver": "gmres",
    #         "error_on_nonconvergence": True,
    #         "relative_tolerance": 1e-5,
    #         "absolute_tolerance": 1e-5,
    #         "maximum_iterations": 20,
    #         "report": False,
    #         "krylov_solver": {
    #             "nonzero_initial_guess": True,
    #             "absolute_tolerance": 1e-10,
    #             "relative_tolerance": 1e-10,
    #             "maximum_iterations": 1000,
    #             "monitor_convergence": False,
    #         },
    #         "lu_solver": {"report": False, "symmetric": False, "verbose": False},
    #     }

    @staticmethod
    def default_solver_parameters():
        return {
            "petsc": {
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
                "ksp_max_it": 500,
                "ksp_rtol": 1e-5,
                "ksp_gmres_restart": 100,
            },
            "newton_verbose": False,
            "ksp_verbose": False,
            "debug": False,
            "linear_solver": "gmres",
            "preconditioner": "hypre_amg",
            "error_on_nonconvergence": True,
            "relative_tolerance": 1e-5,
            "absolute_tolerance": 1e-5,
            "maximum_iterations": 20,
            "report": False,
            "krylov_solver": {
                "nonzero_initial_guess": True,
                "absolute_tolerance": 1e-10,
                "relative_tolerance": 1e-10,
                "maximum_iterations": 1000,
                "monitor_convergence": False,
            },
            "lu_solver": {"report": False, "symmetric": False, "verbose": False},
        }

    # @staticmethod
    # def default_solver_parameters():
    #     return {
    #         "petsc": {
    #             "ksp_type": "preonly",
    #             "ksp_rtol": None,
    #             "ksp_atol": None,
    #             "ksp_max_it": None,
    #             "ksp_norm_type": "preconditioned",
    #             "ksp_gmres_restart": None,
    #             "pc_type": None,
    #             "pc_factor_mat_solver_type": None,
    #             "mat_superlu_dist_equil": True,
    #             "mat_superlu_dist_rowperm": "LargeDiag_MC64",
    #             "mat_superlu_dist_colperm": "PARMETIS",
    #             "mat_superlu_dist_parsymbfact": True,
    #             "mat_superlu_dist_replacetinypivot": True,
    #             "mat_superlu_dist_fact": "DOFACT",
    #             "mat_superlu_dist_iterrefine": True,
    #             "pc_hypre_type": None,
    #         },
    #         "verbose": False,
    #         "linear_solver": "mumps",
    #         "preconditioner": "hypre_amg",
    #         "error_on_nonconvergence": False,
    #         "relative_tolerance": 1e-5,
    #         "absolute_tolerance": 1e-5,
    #         "maximum_iterations": 20,
    #         "report": True,
    #         "krylov_solver": {
    #             "absolute_tolerance": 1e-13,
    #             "relative_tolerance": 1e-13,
    #             "maximum_iterations": 1000,
    #             "monitor_convergence": False,
    #         },
    #         "lu_solver": {"report": False, "symmetric": False, "verbose": False},
    #     }

    def converged(self, r, p, i):
        self._converged_called = True

        res = r.norm("l2")
        logger.debug(f"Mechanics solver residual: {res}")

        if self.debug:
            if not hasattr(self, "_datacollector"):
                logger.debug("No datacollector registered with the NewtonSolver")

            else:
                self._residuals.append(res)

        return super().converged(r, p, i)

    def save_residuals(self) -> None:
        if not self.debug:
            # Only used in debug mode
            return
        if not hasattr(self, "_datacollector"):
            logger.debug("No datacollector registered with the NewtonSolver")

        else:
            self._datacollector.save_residual(
                self._residuals,
                index=self._residual_index,
            )
            self._residual_index += 1
            self._residuals.clear()

    def solver_setup(self, A, J, p, i):
        self._solver_setup_called = True
        super().solver_setup(A, J, p, i)

    def solve(self):
        logger.debug("Solving mechanics")
        self._solve_called = True
        ret = super().solve(self._problem, self._state.vector())
        self._state.vector().apply("insert")
        logger.debug("Done solving mechanics")
        self.save_residuals()
        return ret

    # DEBUGGING
    # This is just to check if we are using the overloaded functions
    def check_overloads_called(self):
        assert getattr(self, "_converged_called", False)
        assert getattr(self, "_solver_setup_called", False)
        assert getattr(self, "_update_solution_called", False)
        assert getattr(self, "_solve_called", False)


class MechanicsNewtonSolver_ODE(MechanicsNewtonSolver):
    # def update_solution(self, x, dx, rp, p, i):
    #     self._update_solution_called = True
    #
    #     import dolfin as _d
    #     if _d.MPI.rank(_d.MPI.comm_world) == 0:
    #         print(f"[DX] iter={i} |dx|={dx.norm('l2'):.6e} |x_before|={x.norm('l2'):.6e}", flush=True)
    #
    #     # Update x from the dx obtained from linear solver (Newton iteration) :
    #     # x = -rp*dx (rp : relax param)
    #     super().update_solution(x, dx, rp, p, i)
    #
    #     # Updating form of MechanicsProblem (from current lmbda, zetas, zetaw, ...)
    #     self._state.vector().set_local(x)
    #     # self._mech_problem._init_forms()
    #     # Recompute Zetas, Zetaw, Ta, lmbda
    #     # self._mech_problem.material.active.update_prev()
    #     self._update_cb()
    #     # Re-init this solver with the new problem (note : done in _init_forms)
    #     # self.__init__(self._mech_problem)

    def update_solution(self, x, dx, rp, p, i):
        self._update_solution_called = True
        super().update_solution(x, dx, rp, p, i)
        self._state.vector().set_local(x)
        self._state.vector().apply("insert")

        import dolfin
        import os
        u = self._state.split(deepcopy=True)[0]
        F = dolfin.Identity(3) + dolfin.grad(u)
        DG0 = dolfin.FunctionSpace(self._state.function_space().mesh(), "DG", 0)
        Jp_fn = dolfin.project(dolfin.det(F), DG0)
        Jp = Jp_fn.vector().get_local()
        rank = dolfin.MPI.rank(dolfin.MPI.comm_world)
        comm = self._state.function_space().mesh().mpi_comm()

        local_bad = 1 if (len(Jp) and (Jp <= 0).any()) else 0
        any_bad = dolfin.MPI.max(comm, local_bad)  # collective: does ANY rank have a bad element?

        if local_bad:
            idx = Jp.argmin()
            c = dolfin.Cell(self._state.function_space().mesh(), idx).midpoint()
            print(
                f"[detF] iter{i} rank{rank}: min={Jp.min():.3e} n_bad={(Jp <= 0).sum()} worst@({c.x():.1f},{c.y():.1f},{c.z():.1f})",
                flush=True)

        if any_bad:
            # ALL ranks must enter together, regardless of local_bad
            dump_dir = os.environ.get("DETF_DUMP_DIR", "detf_dumps")
            if rank == 0:
                os.makedirs(dump_dir, exist_ok=True)
            dolfin.MPI.barrier(comm)
            dump_path = os.path.join(dump_dir, f"detf_fail_iter{i}.h5")
            with dolfin.HDF5File(comm, dump_path, "w") as f:
                f.write(self._state.function_space().mesh(), "mesh")
                f.write(u, "u")
                f.write(Jp_fn, "detF")
            if rank == 0:
                print(f"[detF] dumped state to {dump_path}", flush=True)

        # # I1 check - unconditional, every rank, every iteration
        # C = F.T * F
        # I1 = dolfin.tr(C)
        # I1p = dolfin.project(I1, DG0).vector().get_local()
        #
        # idx_bad = I1p.argmax()
        # cell = dolfin.Cell(self._state.function_space().mesh(), idx_bad)
        # verts = cell.entities(0)
        # V_disp = self._state.function_space().sub(0).collapse()
        # u_fn = dolfin.project(u, V_disp)
        # u_vals = u_fn.compute_vertex_values().reshape(3, -1).T
        # print(f"[I1-detail] iter{i} rank{rank}: cell {idx_bad} vertex displacements:")
        # for v in verts:
        #     coord = self._state.function_space().mesh().coordinates()[v]
        #     print(f"  vertex {v} at {coord}: u={u_vals[v]}")
        #
        # if len(I1p):
        #     idx1 = I1p.argmax()
        #     c1 = dolfin.Cell(self._state.function_space().mesh(), idx1).midpoint()
        #     print(
        #         f"[I1] iter{i} rank{rank}: max={I1p.max():.3e} "
        #         f"worst@({c1.x():.1f},{c1.y():.1f},{c1.z():.1f})",
        #         flush=True)

        self._update_cb()