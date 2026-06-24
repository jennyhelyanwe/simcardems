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
        self.petsc_solver = dolfin.PETScKrylovSolver()
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
        self.linear_solver().set_from_options()
        self._residual_index = 0
        self._residuals = []

    def register_datacollector(self, datacollector):
        self._datacollector = datacollector

    @staticmethod
    def default_solver_parameters():
        return {
            "petsc": {
                "ksp_type": "gmres",
                "ksp_gmres_restart": 100,
                "ksp_rtol": 1e-8,
                "ksp_atol": 1e-10,
                "ksp_max_it": 500,
                "pc_type": "gamg",
                "pc_gamg_type": "agg",
                "pc_gamg_threshold": 0.02,
                "mg_levels_ksp_type": "chebyshev",
                "mg_levels_pc_type": "jacobi",
                "ksp_monitor_true_residual": "",
            },
            "linear_solver": "gmres",
            "preconditioner": "gamg",
            "newton_verbose": False,
            "ksp_verbose": False,
            "debug": False,
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
            "error_on_nonconvergence": True,
            },
            "lu_solver": {"report": False, "symmetric": False, "verbose": False},
        }

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

        if not getattr(self, "_gamg_configured", False):
            import dolfin
            from petsc4py import PETSc

            W = self._state.function_space()
            Vd = W.sub(0).collapse()
            gdim = W.mesh().geometry().dim()

            builders = []
            for k in range(gdim):
                vec = [0.0] * gdim;
                vec[k] = 1.0
                builders.append(dolfin.Constant(tuple(vec)))
            if gdim == 3:
                for e in [("0.0", "-x[2]", "x[1]"),
                          ("x[2]", "0.0", "-x[0]"),
                          ("-x[1]", "x[0]", "0.0")]:
                    builders.append(dolfin.Expression(e, degree=1))

            mode_vecs = []
            for expr in builders:
                d = dolfin.Function(Vd);
                d.interpolate(expr)
                m = dolfin.Function(W);
                dolfin.assign(m.sub(0), d)
                mode_vecs.append(dolfin.as_backend_type(m.vector()).vec().copy())

            # orthonormalise via Gram-Schmidt (GAMG wants an orthonormal basis)
            for a in range(len(mode_vecs)):
                for b in range(a):
                    dot = mode_vecs[a].dot(mode_vecs[b])
                    mode_vecs[a].axpy(-dot, mode_vecs[b])
                nrm = mode_vecs[a].norm()
                if nrm > 0:
                    mode_vecs[a].scale(1.0 / nrm)

            nsp = PETSc.NullSpace().create(
                constant=False, vectors=mode_vecs, comm=W.mesh().mpi_comm()
            )
            dolfin.as_backend_type(A).mat().setNearNullSpace(nsp)
            self._rbm_nsp = nsp  # keep alive — don't let it get GC'd
            self._gamg_configured = True

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
    def update_solution(self, x, dx, rp, p, i):
        self._update_solution_called = True

        # Update x from the dx obtained from linear solver (Newton iteration) :
        # x = -rp*dx (rp : relax param)
        super().update_solution(x, dx, rp, p, i)

        # Updating form of MechanicsProblem (from current lmbda, zetas, zetaw, ...)
        self._state.vector().set_local(x)
        # self._mech_problem._init_forms()
        # Recompute Zetas, Zetaw, Ta, lmbda
        # self._mech_problem.material.active.update_prev()
        self._update_cb()
        # Re-init this solver with the new problem (note : done in _init_forms)
        # self.__init__(self._mech_problem)
