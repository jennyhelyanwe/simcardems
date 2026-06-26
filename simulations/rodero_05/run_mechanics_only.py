import os
cache_dir = os.environ.get("FENICS_CACHE_DIR", os.path.expanduser("~/.cache"))
os.environ["XDG_CACHE_HOME"] = cache_dir

import dolfin
import pulse
import numpy as np
from cardiac_geometries.geometry import Geometry
from simcardems.bivgeometry import BiVentricularGeometry
from simcardems import boundary_conditions

def mpi_print(*args, **kwargs):
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print(*args, **kwargs, flush=True)

# ── Load geometry ──────────────────────────────────────────────────────────────
mpi_print("Loading geometry...")
geo = Geometry.from_file("rodero_05_fine.h5")
biv_geo = BiVentricularGeometry.from_geometry(
    geo, ep_mesh=geo.mesh, ffun_ep=geo.ffun,
)
mpi_print(f"Mesh elements: {biv_geo.mechanics_mesh.num_cells()}")
mpi_print(f"Mesh vertices: {biv_geo.mechanics_mesh.num_vertices()}")

# ── Material ───────────────────────────────────────────────────────────────────
mpi_print("Setting up material...")
material_parameters = pulse.Guccione.default_parameters()
material_parameters["CC"] = 2.0
material_parameters["bf"] = 8.0
material_parameters["bfs"] = 4.0
material_parameters["bt"] = 2.0

material = pulse.Guccione(
    parameters=material_parameters,
    f0=biv_geo.microstructure.f0,
    s0=biv_geo.microstructure.s0,
    n0=biv_geo.microstructure.n0,
)

# ── Boundary conditions ────────────────────────────────────────────────────────
mpi_print("Setting up boundary conditions...")
# ── Boundary conditions ────────────────────────────────────────────────────────
mpi_print("Setting up boundary conditions...")
# lv_pressure = dolfin.Constant(0.0)
# rv_pressure = dolfin.Constant(0.0)
#
# bcs = pulse.BoundaryConditions(
#     dirichlet=(
#         lambda W: dolfin.DirichletBC(
#             W.sub(0),
#             dolfin.Constant((0, 0, 0)),
#             biv_geo.ffun,
#             biv_geo.markers["BASE"][0],
#         ),
#     ),
#     neumann=[
#         pulse.NeumannBC(traction=lv_pressure, marker=biv_geo.markers["ENDO_LV"][0]),
#         pulse.NeumannBC(traction=rv_pressure, marker=biv_geo.markers["ENDO_RV"][0]),
#     ],
#     robin=[],
# )
bcs = boundary_conditions.create_biv_boundary_conditions(
    geo=biv_geo,
    traction_lv=0.0,
    traction_rv=0.0,
    spring=10.0,
)

# Fish out the actual traction constants wired into the form
lv_pressure = bcs.neumann[0].traction
rv_pressure = bcs.neumann[1].traction

# ── Problem ────────────────────────────────────────────────────────────────────
mpi_print("Setting up problem...")

# class BiVMechanicsProblem(pulse.MechanicsProblem):
#     # Default MUMPS requires high memory for factorisation and needs highmem node on ARCHER2.
#     def _set_dirichlet_bc(self):
#         self._dirichlet_bc = []
#         super()._set_dirichlet_bc()

class BiVMechanicsProblem(pulse.MechanicsProblem):
    # Does not require highmem
    def _set_dirichlet_bc(self):
        self._dirichlet_bc = []
        super()._set_dirichlet_bc()

    def _init_solver(self):
        self._problem = pulse.solver.NonlinearProblem(
            J=self._jacobian,
            F=self._virtual_work,
            bcs=self._dirichlet_bc,
        )
        self.solver = pulse.solver.NonlinearSolver(
            self._problem,
            self.state,
            parameters={
                "petsc": {
                    "ksp_type": "gmres",
                    "pc_type": "hypre",
                    "pc_hypre_type": "boomeramg",
                    "ksp_rtol": 1e-6,
                    "ksp_max_it": 500,
                    "ksp_gmres_restart": 30,
                },
                "linear_solver": "gmres",
                "preconditioner": "hypre_amg",
                "error_on_nonconvergence": False,
                "relative_tolerance": 1e-5,
                "absolute_tolerance": 1e-5,
                "maximum_iterations": 20,
                "report": True,
            },
        )

dolfin.PETScOptions.set("snes_monitor")

problem = BiVMechanicsProblem(biv_geo, material, bcs)

# ── Initial solve at zero load ─────────────────────────────────────────────────
mpi_print("Initial solve (zero load)...")
problem.solve()
mpi_print("Initial solve done.")

# ── Apply pressure incrementally via pulse.iterate ────────────────────────────
# ── Apply pressure incrementally via pulse.iterate ────────────────────────────
mpi_print("Applying 0.5 kPa LV pressure via pulse.iterate...")
mpi_print(f"Before iterate - lv_pressure: {float(lv_pressure):.6f}")
mpi_print(f"Before iterate - NeumannBC traction: {float(bcs.neumann[0].traction):.6f}")

pulse.iterate.iterate(
    problem,
    control=(lv_pressure, rv_pressure),
    target=(0.5, 0.17),
)

mpi_print(f"After iterate - lv_pressure: {float(lv_pressure):.6f}")
mpi_print(f"After iterate - NeumannBC traction: {float(bcs.neumann[0].traction):.6f}")
mpi_print("Pressure solve done.")
u, p = problem.state.split(deepcopy=True)
mpi_print(f"State vector norm: {problem.state.vector().norm('l2'):.6e}")
mpi_print(f"Max displacement: {u.vector().norm('linf'):.6e} mm")