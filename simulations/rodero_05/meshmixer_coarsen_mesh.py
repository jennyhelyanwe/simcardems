"""
Coarse mesh generation pipeline for rodero_05.
Part 1 (not included): Meshmixer remeshing of rodero_05 surface to 4mm -> rodero_05_4mm.obj

Meshmixer instructions:
Load in fine surface mesh extract from rodero_05 using paraview Extract Surface filter, as rodero_05_fine.obj

Analysis -> Inspector -> Auto Repair All

Ctrl A to select entire mesh

Edit -> Remesh

Target Edge Length -> 0.2 (the mesh is actually in cm, so 0.2 gives you 2 mm in reality).

Regularity: 100 (make sure triangles are equilateral as much as possible)
Threshold: 50
Iterations: 15
Boundary Mode: Free Boundary


View -> Show Wireframe (to visually check quality).

Accept

Export rodero_05_2_mm.obj

Steps 2-8:
    2. TetGen tetrahedralise with cavity holes, scale to mm
    3. Extract surface, nearest-neighbour marker transfer from fine mesh boundary facets
    4. Build dolfin mesh from VTU, assign ffun from markers
    5. Save to rodero_05_coarse_2mm.h5 with mesh, ffun, marker metadata
    6. KD-tree fibre interpolation from fine mesh
    7. Save fibres to h5
    8. Export XDMF for visualisation
"""

import numpy as np
import pyvista as pv
import tetgen
import dolfin
import h5py
import meshio
from scipy.spatial import cKDTree

# ── Step 2: TetGen tetrahedralisation ────────────────────────────────────────

print("Step 2: Tetrahedralising with TetGen...")

resolution = '2mm'
surf = pv.read("rodero_05_"+resolution+".obj")
bodies = surf.split_bodies()
bodies_sorted = sorted(bodies, key=lambda b: b.area, reverse=True)
epi, endo1, endo2 = bodies_sorted

def nudge_toward(point, target, factor=0.1):
    return np.array(point) + factor * (np.array(target) - np.array(point))

hole1 = nudge_toward(endo1.center, epi.center)
hole2 = nudge_toward(endo2.center, epi.center)
print(f"  Hole points: {hole1}, {hole2}")

tet = tetgen.TetGen(surf)
tet.add_hole(hole1)
tet.add_hole(hole2)
# tet.tetrahedralize(switches="pq1.414/20")
tet.tetrahedralize(switches="pq1.2/15a50")

coarse_pv = tet.grid
coarse_pv.points *= 10  # scale to mm
coarse_pv.save("rodero_05_coarse_"+resolution+".vtu")
print(f"  TetGen mesh: {coarse_pv.n_points} nodes, {coarse_pv.n_cells} cells")

## MMG3D mesh optimisation and quality check
import subprocess
import meshio

# Convert TetGen VTU to .mesh for mmg3d
meshio.write("rodero_05_coarse_tmp.mesh", meshio.read("rodero_05_coarse_"+resolution+".vtu"))

# Run mmg3d optimisation
subprocess.run([
    "mmg3d_O3",
    "-in", "rodero_05_coarse_tmp.mesh",
    "-out", "rodero_05_coarse_opt.mesh",
    "-hmax", "2.0", "-hmin", "1.8", "-hgrad", "1.3", "-hausd", "0.5",
    "-optim"
], check=True)

# Read back optimised mesh
m_opt = meshio.read("rodero_05_coarse_opt.mesh")
coarse_pv = pv.wrap(m_opt)  # continue pipeline with optimised mesh
coarse_pv.save("rodero_05_coarse_"+resolution+".vtu")

# ── Step 3: Nearest-neighbour marker transfer from fine mesh ──────────────────

print("Step 3: Transferring boundary markers from fine mesh...")

fine_mesh = dolfin.Mesh()
with dolfin.HDF5File(fine_mesh.mpi_comm(), "rodero_05_fine.h5", "r") as f:
    f.read(fine_mesh, "mesh", False)
    ffun_fine = dolfin.MeshFunction("size_t", fine_mesh, 2)
    f.read(ffun_fine, "meshfunctions/ffun")

fine_mesh.init()
fine_mesh.init(2, 3)

fine_boundary_centroids = []
fine_boundary_markers = []
for facet in dolfin.facets(fine_mesh):
    if facet.exterior():
        fine_boundary_centroids.append(facet.midpoint().array())
        fine_boundary_markers.append(ffun_fine[facet.index()])

fine_boundary_centroids = np.array(fine_boundary_centroids)
fine_boundary_markers = np.array(fine_boundary_markers)

coords = fine_mesh.coordinates()
cells_arr = fine_mesh.cells()
fine_cell_centres = coords[cells_arr].mean(axis=1)
print(f"  Fine marker counts: { {m: np.sum(fine_boundary_markers==m) for m in np.unique(fine_boundary_markers)} }")

tree_fine = cKDTree(fine_boundary_centroids)

coarse_surf = coarse_pv.extract_surface(algorithm='dataset_surface')
coarse_face_centres = coarse_surf.cell_centers().points
_, idx = tree_fine.query(coarse_face_centres)
coarse_markers = fine_boundary_markers[idx]
print(f"  Coarse marker counts: { {m: np.sum(coarse_markers==m) for m in np.unique(coarse_markers)} }")

# Build KD tree from coarse face centres for dolfin facet assignment
tree_coarse = cKDTree(coarse_face_centres)

# ── Step 4: Build dolfin mesh and assign ffun ─────────────────────────────────

print("Step 4: Building dolfin mesh and assigning boundary markers...")

m = meshio.read("rodero_05_coarse_"+resolution+".vtu")
coords = m.points
cells = m.cells_dict["tetra"]

dolfin_mesh = dolfin.Mesh()
editor = dolfin.MeshEditor()
editor.open(dolfin_mesh, "tetrahedron", 3, 3)
editor.init_vertices(len(coords))
editor.init_cells(len(cells))
for i, pt in enumerate(coords):
    editor.add_vertex(i, pt)
for i, cell in enumerate(cells):
    editor.add_cell(i, cell)
editor.close()
dolfin_mesh.init()
dolfin_mesh.init(2, 3)

ffun = dolfin.MeshFunction("size_t", dolfin_mesh, 2, 0)
for facet in dolfin.facets(dolfin_mesh):
    if facet.exterior():
        mp = facet.midpoint().array()
        _, i = tree_coarse.query(mp)
        ffun[facet.index()] = coarse_markers[i]

print(f"  Dolfin marker counts: { {m: np.sum(ffun.array()==m) for m in np.unique(ffun.array())} }")
print(f"  Mesh: {dolfin_mesh.num_vertices()} vertices, {dolfin_mesh.num_cells()} cells")

# ── Step 5: Save mesh and ffun to h5 ─────────────────────────────────────────

print("Step 5: Saving mesh to rodero_05_coarse_"+resolution+".h5...")

with dolfin.HDF5File(dolfin_mesh.mpi_comm(), "rodero_05_coarse_"+resolution+".h5", "w") as f:
    f.write(dolfin_mesh, "mesh")
    f.write(ffun, "meshfunctions/ffun")

with h5py.File("rodero_05_coarse_"+resolution+".h5", "a") as f:
    f.create_dataset("info/mesh_type", data="biv_ellipsoid")
    f.create_dataset("info/num_refinements", data=0)
    f.create_dataset("markers/BASE",    data=[10, 2])
    f.create_dataset("markers/ENDO_RV", data=[20, 2])
    f.create_dataset("markers/ENDO_LV", data=[30, 2])
    f.create_dataset("markers/EPI",     data=[40, 2])

print("  Saved mesh and ffun.")

# ── Step 6 & 7: Fibre interpolation from fine mesh ────────────────────────────

print("Step 6: Interpolating fibres from fine mesh...")

# Read fine mesh fibres as P1 (how they were written by build_microstructure)
VFS_fine_p1 = dolfin.VectorFunctionSpace(fine_mesh, "P", 1)
f0_fine = dolfin.Function(VFS_fine_p1)
s0_fine = dolfin.Function(VFS_fine_p1)
n0_fine = dolfin.Function(VFS_fine_p1)

with dolfin.HDF5File(fine_mesh.mpi_comm(), "rodero_05_fine.h5", "r") as f:
    f.read(f0_fine, "microstructure/f0")
    f.read(s0_fine, "microstructure/s0")
    f.read(n0_fine, "microstructure/n0")

# Extract nodal values using vertex_to_dof_map to handle DOF ordering correctly
v2d = dolfin.vertex_to_dof_map(VFS_fine_p1)
fine_topo = fine_mesh.cells()  # (n_cells, 4)
n_fine_vertices = fine_mesh.num_vertices()

# def p1_to_cell_centres(fn):
#     """Average P1 nodal values over the 4 corner nodes of each tet."""
#     raw = fn.vector().get_local()
#     nodes = np.zeros((n_fine_vertices, 3))
#     for i in range(n_fine_vertices):
#         nodes[i] = raw[v2d[3*i:3*i+3]]
#     return nodes[fine_topo].mean(axis=1)  # (n_cells, 3)

def p1_to_cell_centres_nearest(fn, fine_mesh, fine_topo, v2d, n_fine_vertices):
    """Assign each fine cell the fiber value at its nearest vertex (no averaging)."""
    raw = fn.vector().get_local()
    nodes = np.zeros((n_fine_vertices, 3))
    for i in range(n_fine_vertices):
        nodes[i] = raw[v2d[3*i:3*i+3]]

    vertex_coords = fine_mesh.coordinates()
    cell_centres = vertex_coords[fine_topo].mean(axis=1)  # still fine to average *positions*
    # pick the vertex among the cell's 4 corners closest to the cell centroid
    corner_coords = vertex_coords[fine_topo]  # (n_cells, 4, 3)
    dists = np.linalg.norm(corner_coords - cell_centres[:, None, :], axis=2)
    nearest_local = np.argmin(dists, axis=1)  # (n_cells,) index into 0..3
    nearest_global = fine_topo[np.arange(len(fine_topo)), nearest_local]
    return nodes[nearest_global]

f0_vals = p1_to_cell_centres_nearest(f0_fine, fine_mesh, fine_topo, v2d, n_fine_vertices)
s0_vals = p1_to_cell_centres_nearest(s0_fine, fine_mesh, fine_topo, v2d, n_fine_vertices)
n0_vals = p1_to_cell_centres_nearest(n0_fine, fine_mesh, fine_topo, v2d, n_fine_vertices)

print(f"  Fine mesh zero f0: {np.sum(np.linalg.norm(f0_vals, axis=1) < 1e-10)} / {len(f0_vals)}")

# Build KD-tree from fine cell centres and query with coarse cell centres
coords = fine_mesh.coordinates()
cells_arr = fine_mesh.cells()
fine_cell_centres = coords[cells_arr].mean(axis=1)
tree_fine = cKDTree(fine_cell_centres)

coords = dolfin_mesh.coordinates()
cells_arr = dolfin_mesh.cells()
coarse_cell_centres = coords[cells_arr].mean(axis=1)
# coarse_cell_centres = np.array([cell.midpoint().array() for cell in dolfin.cells(dolfin_mesh)])
_, idx = tree_fine.query(coarse_cell_centres)

print(f"  Fine cell centres range: {fine_cell_centres.min():.3f} to {fine_cell_centres.max():.3f}")
print(f"  Coarse cell centres range: {coarse_cell_centres.min():.3f} to {coarse_cell_centres.max():.3f}")

# Assign to coarse DG0 functions
# Assign to coarse DG0 functions
VFS_coarse = dolfin.VectorFunctionSpace(dolfin_mesh, "DG", 0)
f0_coarse = dolfin.Function(VFS_coarse)
s0_coarse = dolfin.Function(VFS_coarse)
n0_coarse = dolfin.Function(VFS_coarse)

f0_raw = f0_vals[idx]
s0_raw = s0_vals[idx]
n0_raw = n0_vals[idx]

print("Before Gram-Schmidt:")
dot_fs = np.abs(np.sum(f0_raw * s0_raw, axis=1))
dot_fn = np.abs(np.sum(f0_raw * n0_raw, axis=1))
dot_sn = np.abs(np.sum(s0_raw * n0_raw, axis=1))
print(f"  max|f0.s0|={dot_fs.max():.3e}, max|f0.n0|={dot_fn.max():.3e}, max|s0.n0|={dot_sn.max():.3e}")

def gram_schmidt_orthonormalize(f0_arr, s0_arr, n0_arr):
    """
    Gram-Schmidt orthonormalization of (f0, s0, n0) triads, row-wise.
    f0 is treated as the anchor (kept as-is, just renormalized).
    s0 is orthogonalized against f0, then renormalized.
    n0 is set to f0 x s0 (guarantees exact orthogonality to both), renormalized.
    """
    # Step 1: normalize f0
    f0_norm = np.linalg.norm(f0_arr, axis=1, keepdims=True)
    f0_new = f0_arr / np.where(f0_norm > 1e-10, f0_norm, 1.0)

    # Step 2: orthogonalize s0 against f0, then normalize
    proj = np.einsum('ij,ij->i', s0_arr, f0_new)[:, None] * f0_new
    s0_orth = s0_arr - proj
    s0_norm = np.linalg.norm(s0_orth, axis=1, keepdims=True)
    degenerate_s0 = (s0_norm < 1e-8).flatten()
    if degenerate_s0.sum() > 0:
        print(f"  WARNING: {degenerate_s0.sum()} dofs have s0 nearly parallel to f0 after orthogonalization!")
    s0_new = s0_orth / np.where(s0_norm > 1e-10, s0_norm, 1.0)

    # Step 3: n0 = f0 x s0 (guarantees exact orthogonality to both by construction)
    n0_new = np.cross(f0_new, s0_new)
    n0_norm = np.linalg.norm(n0_new, axis=1, keepdims=True)
    n0_new = n0_new / np.where(n0_norm > 1e-10, n0_norm, 1.0)

    return f0_new, s0_new, n0_new
f0_gs, s0_gs, n0_gs = gram_schmidt_orthonormalize(f0_raw, s0_raw, n0_raw)

print("After Gram-Schmidt:")
dot_fs = np.abs(np.sum(f0_gs * s0_gs, axis=1))
dot_fn = np.abs(np.sum(f0_gs * n0_gs, axis=1))
dot_sn = np.abs(np.sum(s0_gs * n0_gs, axis=1))
print(f"  max|f0.s0|={dot_fs.max():.3e}, max|f0.n0|={dot_fn.max():.3e}, max|s0.n0|={dot_sn.max():.3e}")

f0_coarse.vector().set_local(f0_gs.flatten())
s0_coarse.vector().set_local(s0_gs.flatten())
n0_coarse.vector().set_local(n0_gs.flatten())

print("Step 7: Saving fibres to h5...")

with dolfin.HDF5File(dolfin_mesh.mpi_comm(), "rodero_05_coarse_"+resolution+".h5", "a") as f:
    f.write(f0_coarse, "microstructure/f0")
    f.write(s0_coarse, "microstructure/s0")
    f.write(n0_coarse, "microstructure/n0")

print("  Fibres saved.")

# ── Quality check: fibre orthonormality ──────────────────────────────────────
print("Quality checking fibres...")

with dolfin.HDF5File(dolfin_mesh.mpi_comm(), "rodero_05_coarse_"+resolution+".h5", "r") as f:
    f0_check = dolfin.Function(dolfin.VectorFunctionSpace(dolfin_mesh, "DG", 0))
    s0_check = dolfin.Function(dolfin.VectorFunctionSpace(dolfin_mesh, "DG", 0))
    n0_check = dolfin.Function(dolfin.VectorFunctionSpace(dolfin_mesh, "DG", 0))
    f.read(f0_check, "microstructure/f0")
    f.read(s0_check, "microstructure/s0")
    f.read(n0_check, "microstructure/n0")

f0_arr = f0_check.vector().get_local().reshape(-1, 3)
s0_arr = s0_check.vector().get_local().reshape(-1, 3)
n0_arr = n0_check.vector().get_local().reshape(-1, 3)

f0_norms = np.linalg.norm(f0_arr, axis=1)
s0_norms = np.linalg.norm(s0_arr, axis=1)
n0_norms = np.linalg.norm(n0_arr, axis=1)

print(f"  f0 norms: min={f0_norms.min():.6f} max={f0_norms.max():.6f}")
print(f"  s0 norms: min={s0_norms.min():.6f} max={s0_norms.max():.6f}")
print(f"  n0 norms: min={n0_norms.min():.6f} max={n0_norms.max():.6f}")
print(f"  Zero f0: {np.sum(f0_norms < 1e-10)}")
print(f"  Zero s0: {np.sum(s0_norms < 1e-10)}")
print(f"  Zero n0: {np.sum(n0_norms < 1e-10)}")
print(f"  f0·s0 max: {np.abs(np.sum(f0_arr*s0_arr, axis=1)).max():.6f}")
print(f"  f0·n0 max: {np.abs(np.sum(f0_arr*n0_arr, axis=1)).max():.6f}")
print(f"  s0·n0 max: {np.abs(np.sum(s0_arr*n0_arr, axis=1)).max():.6f}")

# Fine mesh coordinate range check
print(f"  Fine cell centres range: {fine_cell_centres.min():.3f} to {fine_cell_centres.max():.3f} (should be in mm)")
print(f"  Coarse cell centres range: {coarse_cell_centres.min():.3f} to {coarse_cell_centres.max():.3f} (should be in mm)")

if np.sum(f0_norms < 1e-10) > 0:
    print(f"  WARNING: {np.sum(f0_norms < 1e-10)} elements have zero fibre norm!")
else:
    print("  All fibre norms OK.")

print(f"Final mesh: {dolfin_mesh.num_cells()} cells, {dolfin_mesh.num_vertices()} vertices")

import dolfin
import numpy as np

mesh = dolfin.Mesh()
with dolfin.HDF5File(mesh.mpi_comm(), "rodero_05_coarse_2mm.h5", "r") as f:
    f.read(mesh, "mesh", False)
coords = mesh.coordinates()
cells_arr = mesh.cells()

def tet_volumes(coords, cells_arr):
    p0 = coords[cells_arr[:, 0]]
    p1 = coords[cells_arr[:, 1]]
    p2 = coords[cells_arr[:, 2]]
    p3 = coords[cells_arr[:, 3]]
    return np.abs(np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0

volumes = tet_volumes(coords, cells_arr)
print(f"Cell volumes: min={volumes.min():.4f}, mean={volumes.mean():.4f}, max={volumes.max():.4f}")
print(f"Negative volumes: {np.sum(volumes < 0)}")
print(f"Very small volumes (<0.01): {np.sum(volumes < 0.01)}")

# Check inradius/circumradius ratio (quality metric)
def tet_quality_ratios(coords, cells_arr):
    """Vectorized inradius/circumradius ratio for tetrahedra."""
    p0 = coords[cells_arr[:, 0]]
    p1 = coords[cells_arr[:, 1]]
    p2 = coords[cells_arr[:, 2]]
    p3 = coords[cells_arr[:, 3]]

    # Volume (already have from tet_volumes, but recompute signed for clarity)
    vol = np.abs(np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0

    # Face areas (4 faces per tet): use 0.5*|cross product| for each triangular face
    def tri_area(a, b, c):
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    A0 = tri_area(p1, p2, p3)  # face opposite p0
    A1 = tri_area(p0, p2, p3)  # face opposite p1
    A2 = tri_area(p0, p1, p3)  # face opposite p2
    A3 = tri_area(p0, p1, p2)  # face opposite p3
    surface_area = A0 + A1 + A2 + A3

    inradius = 3 * vol / surface_area

    # Circumradius via edge lengths (Cayley-Menger-free approach using edge vectors)
    edges = [p1 - p0, p2 - p0, p3 - p0]
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p0, axis=1)
    c = np.linalg.norm(p3 - p0, axis=1)
    a1 = np.linalg.norm(p3 - p2, axis=1)
    b1 = np.linalg.norm(p3 - p1, axis=1)
    c1 = np.linalg.norm(p2 - p1, axis=1)

    # Circumradius formula for a tetrahedron
    num = np.sqrt(
        (a*a1 + b*b1 + c*c1) *
        (a*a1 + b*b1 - c*c1) *
        (a*a1 - b*b1 + c*c1) *
        (-a*a1 + b*b1 + c*c1)
    )
    circumradius = num / (24 * vol)

    return inradius / circumradius

radii = tet_quality_ratios(coords, cells_arr)
print(f"Inradius/circumradius: min={radii.min():.4f}, mean={radii.mean():.4f}")
print(f"Poor quality (ratio < 0.1): {np.sum(radii < 0.1)}")

import meshio
m = meshio.read("rodero_05_coarse_"+resolution+".vtu")
meshio.write(f"rodero_05_coarse_"+resolution+"_pre_mmg.mesh", m)

# ── Step 7b: Export fibre fields to VTU (fine mesh, point data) ─────────────
print("Step 7b: Exporting fine mesh fibres to VTU...")

def export_fine_fibres_vtu(fine_mesh, f0_fine, s0_fine, n0_fine, v2d, n_fine_vertices,
                            out_path="rodero_05_fine_fibres.vtu"):
    coords = fine_mesh.coordinates()
    cells = fine_mesh.cells()
    vtk_cells = np.hstack([np.full((cells.shape[0], 1), 4), cells]).flatten()
    grid = pv.UnstructuredGrid(vtk_cells, np.full(cells.shape[0], pv.CellType.TETRA), coords)

    def get_point_array(fn):
        raw = fn.vector().get_local()
        nodes = np.zeros((n_fine_vertices, 3))
        for i in range(n_fine_vertices):
            nodes[i] = raw[v2d[3*i:3*i+3]]
        return nodes

    f0_pts = get_point_array(f0_fine)
    s0_pts = get_point_array(s0_fine)
    n0_pts = get_point_array(n0_fine)

    grid.point_data["f0"] = f0_pts
    grid.point_data["s0"] = s0_pts
    grid.point_data["n0"] = n0_pts
    grid.point_data["f0_dot_s0"] = np.abs(np.sum(f0_pts * s0_pts, axis=1))
    grid.point_data["f0_dot_n0"] = np.abs(np.sum(f0_pts * n0_pts, axis=1))
    grid.point_data["s0_dot_n0"] = np.abs(np.sum(s0_pts * n0_pts, axis=1))

    grid.save(out_path)
    print(f"  Saved {out_path}")

export_fine_fibres_vtu(fine_mesh, f0_fine, s0_fine, n0_fine, v2d, n_fine_vertices,
                        out_path="rodero_05_fine_fibres.vtu")

# ── Step 7c: Export fibre fields to VTU (coarse mesh, cell data) ────────────
print("Step 7c: Exporting coarse mesh fibres to VTU...")

def export_coarse_fibres_vtu(dolfin_mesh, f0_func, s0_func, n0_func,
                              out_path="rodero_05_coarse_"+resolution+"_fibres.vtu"):
    coords = dolfin_mesh.coordinates()
    cells = dolfin_mesh.cells()
    vtk_cells = np.hstack([np.full((cells.shape[0], 1), 4), cells]).flatten()
    grid = pv.UnstructuredGrid(vtk_cells, np.full(cells.shape[0], pv.CellType.TETRA), coords)

    f0_arr = f0_func.vector().get_local().reshape(-1, 3)
    s0_arr = s0_func.vector().get_local().reshape(-1, 3)
    n0_arr = n0_func.vector().get_local().reshape(-1, 3)

    grid.cell_data["f0"] = f0_arr
    grid.cell_data["s0"] = s0_arr
    grid.cell_data["n0"] = n0_arr
    grid.cell_data["f0_dot_s0"] = np.abs(np.sum(f0_arr * s0_arr, axis=1))
    grid.cell_data["f0_dot_n0"] = np.abs(np.sum(f0_arr * n0_arr, axis=1))
    grid.cell_data["s0_dot_n0"] = np.abs(np.sum(s0_arr * n0_arr, axis=1))

    grid.save(out_path)
    print(f"  Saved {out_path}")

export_coarse_fibres_vtu(dolfin_mesh, f0_coarse, s0_coarse, n0_coarse,
                          out_path="rodero_05_coarse_"+resolution+"_fibres.vtu")

def gram_schmidt_orthonormalize(f0_arr, s0_arr, n0_arr):
    """
    Gram-Schmidt orthonormalization of (f0, s0, n0) triads, row-wise.
    f0 is treated as the anchor (kept as-is direction, just renormalized).
    s0 is orthogonalized against f0, then renormalized.
    n0 is set to f0 x s0 (guarantees exact orthogonality to both), renormalized.
    """
    f0_norm = np.linalg.norm(f0_arr, axis=1, keepdims=True)
    f0_new = f0_arr / np.where(f0_norm > 1e-10, f0_norm, 1.0)

    proj = np.einsum('ij,ij->i', s0_arr, f0_new)[:, None] * f0_new
    s0_orth = s0_arr - proj
    s0_norm = np.linalg.norm(s0_orth, axis=1, keepdims=True)
    degenerate_s0 = (s0_norm < 1e-8).flatten()
    if degenerate_s0.sum() > 0:
        print(f"  WARNING: {degenerate_s0.sum()} elements have s0 nearly parallel to f0 after orthogonalization!")
    s0_new = s0_orth / np.where(s0_norm > 1e-10, s0_norm, 1.0)

    n0_new = np.cross(f0_new, s0_new)
    n0_norm = np.linalg.norm(n0_new, axis=1, keepdims=True)
    n0_new = n0_new / np.where(n0_norm > 1e-10, n0_norm, 1.0)

    return f0_new, s0_new, n0_new



# ── Step 8: Export XDMF for visualisation ────────────────────────────────────
print("Step 8: Exporting XDMF...")

with dolfin.XDMFFile("rodero_05_coarse_mesh_"+resolution+".xdmf") as xf:
    xf.write(dolfin_mesh)

with dolfin.XDMFFile("rodero_05_coarse_boundaries_"+resolution+".xdmf") as xf:
    xf.write(ffun)

print("Done. Output: rodero_05_coarse_"+resolution+".h5, rodero_05_coarse_mesh_"+resolution+".xdmf, rodero_05_coarse_boundaries_"+resolution+".xdmf")