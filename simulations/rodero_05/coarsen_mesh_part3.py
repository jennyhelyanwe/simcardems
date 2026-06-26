import dolfin
import numpy as np
from scipy.spatial import cKDTree

# Load fine mesh with fibres
fine_mesh = dolfin.Mesh()
with dolfin.HDF5File(fine_mesh.mpi_comm(), "rodero_05_fine.h5", "r") as f:
    f.read(fine_mesh, "mesh", False)
    f0_fine = dolfin.Function(dolfin.VectorFunctionSpace(fine_mesh, "DG", 0))
    s0_fine = dolfin.Function(dolfin.VectorFunctionSpace(fine_mesh, "DG", 0))
    n0_fine = dolfin.Function(dolfin.VectorFunctionSpace(fine_mesh, "DG", 0))
    f.read(f0_fine, "microstructure/f0")
    f.read(s0_fine, "microstructure/s0")
    f.read(n0_fine, "microstructure/n0")

coarse_mesh = dolfin.Mesh()
with dolfin.HDF5File(coarse_mesh.mpi_comm(), "rodero_05_coarse.h5", "r") as f:
    f.read(coarse_mesh, "mesh", False)
    ffun_coarse = dolfin.MeshFunction("size_t", coarse_mesh, 2)
    f.read(ffun_coarse, "meshfunctions/ffun")

# Get fine mesh cell centres
fine_cell_centres = np.array([cell.midpoint().array() for cell in dolfin.cells(fine_mesh)])

# Build KD tree
tree = cKDTree(fine_cell_centres)

# Get coarse mesh cell centres
coarse_cell_centres = np.array([cell.midpoint().array() for cell in dolfin.cells(coarse_mesh)])

# Find nearest fine cell for each coarse cell
_, idx = tree.query(coarse_cell_centres)

# Extract fibre values at nearest fine cells
f0_vals = f0_fine.vector().get_local().reshape(-1, 3)
s0_vals = s0_fine.vector().get_local().reshape(-1, 3)
n0_vals = n0_fine.vector().get_local().reshape(-1, 3)

# Create DG0 functions on coarse mesh
DG0 = dolfin.VectorFunctionSpace(coarse_mesh, "DG", 0)
f0_coarse = dolfin.Function(DG0)
s0_coarse = dolfin.Function(DG0)
n0_coarse = dolfin.Function(DG0)

# Set values from nearest neighbour
f0_coarse.vector().set_local(f0_vals[idx].flatten())
s0_coarse.vector().set_local(s0_vals[idx].flatten())
n0_coarse.vector().set_local(n0_vals[idx].flatten())

# Normalise
f0_arr = f0_coarse.vector().get_local().reshape(-1, 3)
s0_arr = s0_coarse.vector().get_local().reshape(-1, 3)
n0_arr = n0_coarse.vector().get_local().reshape(-1, 3)

f0_norm = np.linalg.norm(f0_arr, axis=1, keepdims=True)
s0_norm = np.linalg.norm(s0_arr, axis=1, keepdims=True)
n0_norm = np.linalg.norm(n0_arr, axis=1, keepdims=True)

f0_arr = np.where(f0_norm > 1e-10, f0_arr / f0_norm, f0_arr)
s0_arr = np.where(s0_norm > 1e-10, s0_arr / s0_norm, s0_arr)
n0_arr = np.where(n0_norm > 1e-10, n0_arr / n0_norm, n0_arr)

f0_coarse.vector().set_local(f0_arr.flatten())
s0_coarse.vector().set_local(s0_arr.flatten())
n0_coarse.vector().set_local(n0_arr.flatten())

# Save
with dolfin.HDF5File(coarse_mesh.mpi_comm(), "rodero_05_coarse.h5", "a") as f:
    f.write(f0_coarse, "microstructure/f0")
    f.write(s0_coarse, "microstructure/s0")
    f.write(n0_coarse, "microstructure/n0")

print("Done. Fibres written to rodero_05_coarse.h5")


import dolfin

mesh = dolfin.Mesh()
with dolfin.HDF5File(mesh.mpi_comm(), "rodero_05_coarse.h5", "r") as f:
    f.read(mesh, "mesh", False)
    ffun = dolfin.MeshFunction("size_t", mesh, 2)
    f.read(ffun, "meshfunctions/ffun")

with dolfin.XDMFFile("rodero_05_coarse_mesh.xdmf") as xf:
    xf.write(mesh)

with dolfin.XDMFFile("rodero_05_coarse_boundaries.xdmf") as xf:
    xf.write(ffun)

print("Done")