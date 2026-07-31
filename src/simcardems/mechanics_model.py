from __future__ import annotations

import logging
import typing

import dolfin
import pulse

from . import boundary_conditions
from . import config
from . import geometry
from . import lvgeometry
from . import slabgeometry
from . import bivgeometry
from . import utils
from .newton_solver import MechanicsNewtonSolver
from .newton_solver import MechanicsNewtonSolver_ODE

if typing.TYPE_CHECKING:
    from .models import em_model

logger = utils.getLogger(__name__)


class NearlyIncompressibleHO(pulse.HolzapfelOgden):
    def __init__(self, *args, kappa=1e3, **kwargs):
        self.kappa = kappa
        super().__init__(*args, **kwargs)

    def compressibility(self, p, J):
        return self.kappa / 2.0 * (J - 1.0) ** 2 - p * (J - 1.0)

def setup_solver(
    coupling: em_model.BaseEMCoupling,
    ActiveModel,
    bnd_rigid: bool = config.Config.bnd_rigid,
    pre_stretch: typing.Optional[typing.Union[dolfin.Constant, float]] = None,
    traction: typing.Union[dolfin.Constant, float] = None,
    spring: typing.Union[dolfin.Constant, float] = None,
    fix_right_plane: bool = config.Config.fix_right_plane,
    debug_mode: bool = config.Config.debug_mode,
    set_material: str = "",
    linear_solver="gmres",
    use_custom_newton_solver: bool = config.Config.mechanics_use_custom_newton_solver,
    state_prev=None,
    material_parameters: typing.Optional[dict] = None,
    base_displacement: typing.Optional[dict] = None,
    valve_stiffness_scale: float = 50.0,
):
    """Setup mechanics model with dirichlet boundary conditions or rigid motion."""

    if ActiveModel is None:
        return None
    logger.info("Set up mechanics model")

    if material_parameters is None:
        # Default: Biaxial test parameters from Holzapfel 2019 (Table 1)
        material_parameters = dict(
            a=2.28,
            a_f=1.686,
            b=9.726,
            b_f=15.779,
            a_s=0.0,
            b_s=0.0,
            a_fs=0.0,
            b_fs=0.0,
        )

    active_model = ActiveModel(coupling=coupling, parameters=coupling.cell_params())
    material = pulse.HolzapfelOgden(
        active_model=active_model,
        parameters=material_parameters,
    )
    # print('Using Nearly Incompressible HO')
    # material = NearlyIncompressibleHO(
    #     active_model=active_model,
    #     parameters=material_parameters,
    #     kappa=1e3,
    # )

    if set_material == "Guccione":
        material_parameters = pulse.Guccione.default_parameters()
        material_parameters["CC"] = 2.0
        material_parameters["bf"] = 8.0
        material_parameters["bfs"] = 4.0
        material_parameters["bt"] = 2.0

        material = pulse.Guccione(
            active_model=active_model,
            parameters=material_parameters,
        )
    logger.info("Starting MechanicsProblem init...")
    problem = create_problem(
        material=material,
        geo=coupling.geometry,
        bnd_rigid=bnd_rigid,
        pre_stretch=pre_stretch,
        traction=traction,
        spring=spring,
        fix_right_plane=fix_right_plane,
        linear_solver=linear_solver,
        use_custom_newton_solver=use_custom_newton_solver,
        debug_mode=debug_mode,
        base_displacement=base_displacement,
        valve_stiffness_scale=valve_stiffness_scale,
    )

    if state_prev is not None:
        problem.state.assign(state_prev)

    logger.info("MechanicsProblem init done. Starting initial solve...")
    problem.solve()
    logger.info("Initial mechanics solve done.")
    logger.info("Coupling register model")
    coupling.register_mech_model(problem)
    coupling.print_mechanics_info()

    return problem


class ContinuationBasedMechanicsProblem(pulse.MechanicsProblem):
    def __init__(self, *args, **kwargs):
        self._use_custom_newton_solver = kwargs.pop("use_custom_newton_solver", False)
        self.valve_stiffness_scale = kwargs.pop("valve_stiffness_scale", 50.0)
        super().__init__(*args, **kwargs)
        self.old_states = []
        self.old_controls = []

    def _set_dirichlet_bc(self):
        self._dirichlet_bc = []
        super()._set_dirichlet_bc()

    def _init_forms(self):
        raise NotImplementedError

    def solve(self):
        self._init_forms()
        return super().solve()

    def solve_for_control(self, control, tol=1e-5):
        """Solve with a continuation step for
        better initial guess for the Newton solver
        """
        if len(self.old_controls) >= 2:
            # Find a better initial guess for the solver
            c0, c1 = self.old_controls
            s0, s1 = self.old_states

            denominator = dolfin.assemble((c1 - c0) ** 2 * dolfin.dx)
            # max_denom = self.geometry.mesh.mpi_comm().allreduce(denominator, op=MPI.MAX)

            if denominator > tol:
                numerator = dolfin.assemble((control - c0) ** 2 * dolfin.dx)
                delta = numerator / denominator
                self.state.vector().zero()
                self.state.vector().axpy(1.0 - delta, s0.vector())
                self.state.vector().axpy(delta, s1.vector())

                # Keep track of the newest state
                self.old_controls = [c1]
                self.old_states = [s1]
            else:
                # Keep track of the old state
                self.old_controls = [c0]
                self.old_states = [s0]

        self.solve()

        self.old_states.append(self.state.copy(deepcopy=True))
        self.old_controls.append(control.copy(deepcopy=True))


class MechanicsProblem(ContinuationBasedMechanicsProblem):
    def _init_spaces(self):
        mesh = self.geometry.mesh

        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)

        self.state_space = dolfin.FunctionSpace(
            mesh,
            dolfin.MixedElement([P2, P1]),
        )
        self._init_functions()

    @property
    def strong_coupling(self):
        return hasattr(self.material.active, "Ta")

    @property
    def u_subspace_index(self) -> int:
        return 0

    def _init_functions(self):
        self.state = dolfin.Function(self.state_space, name="state")
        self.state_test = dolfin.TestFunction(self.state_space)

    def _init_forms(self, init_solver: bool = True):
        u, p = dolfin.split(self.state)
        v, q = dolfin.split(self.state_test)

        # Some mechanical quantities
        self._F = dolfin.variable(pulse.DeformationGradient(u))
        self._J = pulse.Jacobian(self._F)
        dx = self.geometry.dx

        valve_mask = getattr(self.geometry, 'valve_mask', None)
        if valve_mask is not None:
            scale_val = getattr(self, 'valve_stiffness_scale', 50.0)
            stiffness_scale = valve_mask * dolfin.Constant(scale_val) + (1.0 - valve_mask) * dolfin.Constant(1.0)
            internal_energy = (
                    stiffness_scale * self.material.strain_energy(self._F)
                    + self.material.compressibility(p, self._J)
            )
        else:
            internal_energy = self.material.strain_energy(self._F) + self.material.compressibility(p, self._J)
        #
        # kappa = dolfin.Constant(500.0)  # bulk modulus (kPa), nearly-incompressible penalty
        # internal_energy = self.material.strain_energy(
        #     self._F,
        # ) + self.material.compressibility(p, self._J) + 0.5 * kappa * (self._J - 1.0) ** 2

        self._virtual_work = dolfin.derivative(
            internal_energy * dx,
            self.state,
            self.state_test,
        )
        # print('self.strong_coupling: ', self.strong_coupling)

        if self.strong_coupling:
            f0 = self.material.active.f0
            f = self._F * f0
            lmbda = dolfin.sqrt(f ** 2)
            # Use frozen Ta_current for both residual and Jacobian to avoid
            # JIT hang from ufl.min_value/ufl.max_value in symbolic Ta(lmbda)
            valve_mask = getattr(self.geometry, 'valve_mask', None)
            if valve_mask is not None:
                myocardium = 1.0 - valve_mask
                Pa_frozen = myocardium * self.material.active.Ta_current * dolfin.outer(f, f0)
            else:
                Pa_frozen = self.material.active.Ta_current * dolfin.outer(f, f0)
            # Pa_frozen = self.material.active.Ta_current * dolfin.outer(f, f0)
            # print('Pa_frozen', Pa_frozen)
            # print('Ta(lmbda):' , self.material.active.Ta(lmbda))
            # print('lmbda: ', lmbda)
            self._virtual_work += dolfin.inner(Pa_frozen, dolfin.grad(v)) * dx
            # print('virtual work add: ', dolfin.inner(Pa_frozen, dolfin.grad(v)) * dx)

        external_work = self._external_work(u, v)
        if external_work is not None:
            self._virtual_work += external_work

        self._set_dirichlet_bc()
        self._jacobian = dolfin.derivative(
            self._virtual_work,
            self.state,
            dolfin.TrialFunction(self.state_space),
        )
        # print('jacobian', self._jacobian)
        # quit()
        if init_solver:
            self._init_solver()

    def _init_solver(self):
        self._problem = pulse.NonlinearProblem(
            J=self._jacobian,
            F=self._virtual_work,
            bcs=self._dirichlet_bc,
        )

        if self._use_custom_newton_solver:
            cls = MechanicsNewtonSolver_ODE
        else:
            cls = MechanicsNewtonSolver

        self.solver = cls(
            problem=self._problem,
            state=self.state,
            update_cb=self.material.active.update_prev,
            parameters=self.solver_parameters,
        )

    def solve(self):
        self._init_forms(init_solver=False)
        newton_iteration, newton_converged = self.solver.solve()
        getattr(self.solver, "check_overloads_called", None)

        if self.strong_coupling:
            u, p = self.state.split(deepcopy=True)

            F = dolfin.grad(u) + dolfin.Identity(3)
            f = F * self.material.active.f0
            lmbda = dolfin.sqrt(f**2)

            self.material.active._projector.project(self.material.active.lmbda, lmbda)
            if self.material.active.dt > 0:
                self.material.active._projector.project(
                    self.material.active._dLambda,
                    (lmbda - self.material.active.lmbda_prev) / self.material.active.dt,
                )
            self.material.active._projector.project(self.material.active.Ta_current, self.material.active.Ta(lmbda))
            self.material.active.update_current(lmbda=lmbda)
            self.material.active.update_prev()

        return newton_iteration, newton_converged


class RigidMotionProblem(MechanicsProblem):
    def _init_spaces(self):
        mesh = self.geometry.mesh

        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P3 = dolfin.VectorElement("Real", mesh.ufl_cell(), 0, 6)

        self.state_space = dolfin.FunctionSpace(
            mesh,
            dolfin.MixedElement([P1, P2, P3]),
        )

        self._init_functions()

    @property
    def u_subspace_index(self) -> int:
        return 1

    def _handle_bcs(self, bcs, bcs_parameters):
        self.bcs = pulse.BoundaryConditions()
        self._dirichlet_bc = []

    def _init_forms(self, init_solver: bool = True):
        # Displacement and hydrostatic_pressure; 3rd space for rigid motion component
        p, u, r = dolfin.split(self.state)
        q, v, z = dolfin.split(self.state_test)

        # Some mechanical quantities
        self._F = dolfin.variable(pulse.DeformationGradient(u))
        self._J = pulse.Jacobian(self._F)
        dx = self.geometry.dx

        internal_energy = self.material.strain_energy(
            self._F,
        ) + self.material.compressibility(p, self._J)

        self._virtual_work = dolfin.derivative(
            internal_energy * dx,
            self.state,
            self.state_test,
        )

        external_work = self._external_work(u, v)
        if external_work is not None:
            self._virtual_work += external_work

        self._virtual_work += dolfin.derivative(
            RigidMotionProblem.rigid_motion_term(
                mesh=self.geometry.mesh,
                u=u,
                r=r,
            ),
            self.state,
            self.state_test,
        )

        f0 = self.material.active.f0
        f = self._F * f0
        lmbda = dolfin.sqrt(f ** 2)
        Pa = self.material.active.Ta(lmbda) * dolfin.outer(f, f0)
        self._virtual_work += dolfin.inner(Pa, dolfin.grad(v)) * dx

        # Use frozen Ta_current for Jacobian — avoids JIT hang from
        # differentiating through ufl.min_value/ufl.max_value in Ta(lmbda)
        valve_mask = getattr(self.geometry, 'valve_mask', None)
        if valve_mask is not None:
            myocardium = 1.0 - valve_mask
            Pa_frozen = myocardium * self.material.active.Ta_current * dolfin.outer(f, f0)
        else:
            Pa_frozen = self.material.active.Ta_current * dolfin.outer(f, f0)
        # Pa_frozen = self.material.active.Ta_current * dolfin.outer(f, f0)
        virtual_work_for_jacobian = (
                self._virtual_work
                - dolfin.inner(Pa, dolfin.grad(v)) * dx
                + dolfin.inner(Pa_frozen, dolfin.grad(v)) * dx
        )
        self._jacobian = dolfin.derivative(
            virtual_work_for_jacobian,
            self.state,
            dolfin.TrialFunction(self.state_space),
        )

        if init_solver:
            self._init_solver()

    def rigid_motion_term(mesh, u, r):
        position = dolfin.SpatialCoordinate(mesh)
        RM = [
            dolfin.Constant((1, 0, 0)),
            dolfin.Constant((0, 1, 0)),
            dolfin.Constant((0, 0, 1)),
            dolfin.cross(position, dolfin.Constant((1, 0, 0))),
            dolfin.cross(position, dolfin.Constant((0, 1, 0))),
            dolfin.cross(position, dolfin.Constant((0, 0, 1))),
        ]

        return sum(dolfin.dot(u, zi) * r[i] * dolfin.dx for i, zi in enumerate(RM))

    def solve(self):
        self._init_forms(init_solver=False)
        newton_iteration, newton_converged = self.solver.solve()
        getattr(self.solver, "check_overloads_called", None)

        if self.strong_coupling:
            p, u, r = self.state.split(deepcopy=True)  # note: p, u, r order

            F = dolfin.grad(u) + dolfin.Identity(3)
            f = F * self.material.active.f0
            lmbda = dolfin.sqrt(f ** 2)

            self.material.active._projector.project(self.material.active.lmbda, lmbda)
            if self.material.active.dt > 0:
                self.material.active._projector.project(
                    self.material.active._dLambda,
                    (lmbda - self.material.active.lmbda_prev) / self.material.active.dt,
                )

            self.material.active._projector.project(self.material.active.Ta_current, self.material.active.Ta(lmbda))
            self.material.active.update_current(lmbda=lmbda)
            self.material.active.update_prev()

        return newton_iteration, newton_converged

def resolve_boundary_conditions(
    geo: geometry.BaseGeometry,
    pre_stretch: typing.Optional[typing.Union[dolfin.Constant, float]] = None,
    traction: typing.Union[dolfin.Constant, float] = None,
    spring: typing.Union[dolfin.Constant, float] = None,
    fix_right_plane: bool = config.Config.fix_right_plane,
    base_displacement: typing.Union[dolfin.Constant, float] = None,
) -> pulse.BoundaryConditions:
    if isinstance(geo, slabgeometry.SlabGeometry):
        return boundary_conditions.create_slab_boundary_conditions(
            geo=geo,
            pre_stretch=pre_stretch,
            traction=traction,
            spring=spring,
            fix_right_plane=fix_right_plane,
        )
    elif isinstance(geo, lvgeometry.LeftVentricularGeometry):
        initial_pressure = 0.001 if traction is None else traction
        return boundary_conditions.create_lv_no_dirichlet_boundary_conditions(
            geo=geo,
            traction=initial_pressure,
            spring=spring,
        )
    elif isinstance(geo, bivgeometry.BiVentricularGeometry):
        # Small nonzero initial pressure -- avoids a degenerate/singular
        # starting solve given no Dirichlet base constraint, before the
        # cycle controller's first real step() call takes over.
        initial_pressure = 0.001 if traction is None else traction
        return boundary_conditions.create_biv_boundary_conditions(
            geo=geo,
            traction_lv=initial_pressure,
            traction_rv=initial_pressure,
            spring=spring,
            base_displacement=base_displacement
        )
    else:
        raise NotImplementedError


def create_problem(
    material: pulse.Material,
    geo: geometry.BaseGeometry,
    bnd_rigid: bool = config.Config.bnd_rigid,
    pre_stretch: typing.Optional[typing.Union[dolfin.Constant, float]] = None,
    traction: typing.Union[dolfin.Constant, float] = None,
    spring: typing.Union[dolfin.Constant, float] = None,
    fix_right_plane: bool = config.Config.fix_right_plane,
    linear_solver="gmres",
    use_custom_newton_solver: bool = config.Config.mechanics_use_custom_newton_solver,
    debug_mode=config.Config.debug_mode,
    base_displacement: typing.Union[dolfin.Constant, float] = None,
    valve_stiffness_scale: float = 50.0,
) -> MechanicsProblem:
    Problem = MechanicsProblem
    if bnd_rigid:
        bcs = None
        Problem = RigidMotionProblem
    else:
        bcs = resolve_boundary_conditions(
            geo=geo,
            pre_stretch=pre_stretch,
            traction=traction,
            spring=spring,
            fix_right_plane=fix_right_plane,
            base_displacement=base_displacement
        )

    verbose = logger.getEffectiveLevel() < logging.INFO
    return Problem(
        geo,
        material,
        bcs,
        solver_parameters={
            "linear_solver": "mumps",
            "petsc": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_33": 0,
            },
            "relative_tolerance": 1e-5,
            "absolute_tolerance": 1e-5,
            "maximum_iterations": 20,
            "report": True,
            "error_on_nonconvergence": False,
        },
        use_custom_newton_solver=use_custom_newton_solver,
        valve_stiffness_scale=valve_stiffness_scale,
    )
