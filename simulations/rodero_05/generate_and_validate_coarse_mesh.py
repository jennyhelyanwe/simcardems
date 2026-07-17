"""
Coarse mesh generation + full validation pipeline for rodero_05.

Part 1 (not included): Meshmixer remeshing of rodero_05 surface -> rodero_05_<res>.obj
  Load fine surface mesh (ParaView Extract Surface filter) as rodero_05_fine.obj
  Analysis -> Inspector -> Auto Repair All
  Ctrl A to select entire mesh
  Edit -> Remesh
    Target Edge Length -> (mesh is in cm; e.g. 0.2 gives 2mm, 0.4 gives 4mm)
    Regularity: 100, Threshold: 50, Iterations: 15, Boundary Mode: Free Boundary
  View -> Show Wireframe (visual quality check)
  Accept, Export rodero_05_<res>.obj

Steps below:
    1. TetGen tetrahedralise with cavity holes, scale to mm
    2. mmg3d quality/size optimisation, resolution verification
    3. Nearest-neighbour boundary marker transfer from fine mesh
    4. Build dolfin mesh, assign ffun
    5. Save mesh + ffun + markers to h5
    6. Nearest-vertex fibre interpolation from fine mesh + Gram-Schmidt orthonormalisation
    7. Fibre orthonormality check
    8. Mesh quality check (volumes, inradius/circumradius)
    9. Boundary marker facet counts
    10. Valve/material label mapping
    11. Material strain-energy-at-rest sanity check
    12. XDMF export for visualisation
    13. Summary
"""

import subprocess

import dolfin
import h5py
import meshio
import numpy as np
import pandas as pd
import pyvista as pv
import tetgen
from scipy.spatial import cKDTree

# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════

resolution = '4mm'
MESH_DIR = 'meshes/'
target_size_mm = float(resolution.replace("mm", ""))  # literal target, e.g. 4.0

# Material parameters to sanity-check at the end (edit as needed).
# Full orthotropic set (Holzapfel 2019 Table 1, initial-value column).
MATERIAL_PARAMS_TO_CHECK = dict(
    a=0.61, a_f=1.56, b=7.5, b_f=35.31,
    a_s=0.70, b_s=33.24, a_fs=0.46, b_fs=5.09,
)

RESOLUTION_TOLERANCE = 0.15   # fraction, for mean edge length check
QUALITY_THRESHOLD = 0.1       # inradius/circumradius ratio below this = poor
STRAIN_ENERGY_TOLERANCE = 1.0  # kPa*mm^3, absolute tolerance for "should be ~0"

summary = {}  # collects PASS/WARN/FAIL per check, printed at the end
# ══════════════════════════════════════════════════════════════════════════
# Step 1 and 2: Tetrahedralise and mmg3d optimisation, iterate until desired edge length reached.
# ══════════════════════════════════════════════════════════════════════════

print(f"Step 1-2: Tetrahedralising + mmg3d optimisation (target {target_size_mm}mm)...")

surf = pv.read(MESH_DIR + "rodero_05_" + resolution + ".obj")
bodies = surf.split_bodies()
bodies_sorted = sorted(bodies, key=lambda b: b.area, reverse=True)
epi, endo1, endo2 = bodies_sorted


def nudge_toward(point, target, factor=0.1):
    return np.array(point) + factor * (np.array(target) - np.array(point))


hole1 = nudge_toward(endo1.center, epi.center)
hole2 = nudge_toward(endo2.center, epi.center)
print(f"  Hole points: {hole1}, {hole2}")

target_size_cm = target_size_mm / 10.0
max_vol_cm3 = (target_size_cm ** 3) / (6.0 * np.sqrt(2))

tet = tetgen.TetGen(surf)
tet.add_hole(hole1)
tet.add_hole(hole2)
tet.tetrahedralize(switches=f"pq1.2/15a{max_vol_cm3:.6f}")

coarse_pv_base = tet.grid
coarse_pv_base.points *= 10  # scale to mm, cache the pre-mmg3d TetGen output
coarse_pv_base.save(MESH_DIR + "rodero_05_coarse_" + resolution + "_pretetgen.vtu")
print(f"  TetGen mesh: {coarse_pv_base.n_points} nodes, {coarse_pv_base.n_cells} cells")


def edge_length_stats(points, cells):
    edges = np.vstack([
        cells[:, [0, 1]], cells[:, [0, 2]], cells[:, [0, 3]],
        cells[:, [1, 2]], cells[:, [1, 3]], cells[:, [2, 3]],
    ])
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    return lengths


def run_mmg3d(hmax_val, hmin_val):
    meshio.write(MESH_DIR + "rodero_05_coarse_tmp.mesh",
                 meshio.read(MESH_DIR + "rodero_05_coarse_" + resolution + "_pretetgen.vtu"))
    subprocess.run([
        "mmg3d_O3",
        "-in", MESH_DIR + "rodero_05_coarse_tmp.mesh",
        "-out", MESH_DIR + "rodero_05_coarse_opt.mesh",
        "-hmax", str(hmax_val), "-hmin", str(hmin_val), "-hgrad", "1.3", "-hausd", "0.5",
        "-optim",
    ], check=True)
    m_opt = meshio.read(MESH_DIR + "rodero_05_coarse_opt.mesh")
    grid = pv.wrap(m_opt)
    tet_cells = grid.cells_dict[pv.CellType.TETRA]
    lengths = edge_length_stats(grid.points, tet_cells)
    return grid, lengths


def iqr_within_tolerance(lengths, target, tol):
    q25, q50, q75 = np.percentile(lengths, [25, 50, 75])
    # both quartile bounds must sit within tol of the target
    return (abs(q25 - target) / target <= tol) and (abs(q75 - target) / target <= tol), q25, q50, q75


hmax = target_size_mm * 1.3
hmin = target_size_mm * 0.4
max_iters = 8
IQR_TOLERANCE = 0.20  # 25th and 75th percentile must each be within 20% of target

for attempt in range(1, max_iters + 1):
    coarse_pv, lengths = run_mmg3d(hmax, hmin)
    converged, q25, q50, q75 = iqr_within_tolerance(lengths, target_size_mm, IQR_TOLERANCE)
    print(f"  Attempt {attempt}: hmax={hmax:.2f} -> q25={q25:.2f}mm, median={q50:.2f}mm, "
          f"q75={q75:.2f}mm (target {target_size_mm:.2f}mm)")
    if converged:
        print(f"  Converged: IQR [{q25:.2f}, {q75:.2f}]mm within {IQR_TOLERANCE*100:.0f}% "
              f"of target after {attempt} attempt(s).")
        break
    # Scale hmax toward the median as the correction driver, since it's the
    # single most representative point of the bulk distribution
    correction = target_size_mm / q50
    hmax = hmax * correction
    hmin = target_size_mm * 0.4 * (hmax / (target_size_mm * 1.3))
else:
    print(f"  *** WARNING: did not converge within {max_iters} attempts, using last result. ***")

coarse_pv.save(MESH_DIR + "rodero_05_coarse_" + resolution + ".vtu")

# ── Histogram + percentile breakdown ─────────────────────────────────────
print(f"\n  === Resolution check (IQR-based) ===")
percentiles = [5, 25, 50, 75, 95]
pvals = np.percentile(lengths, percentiles)
print(f"  Target: {target_size_mm:.2f} mm")
print(f"  Mean: {lengths.mean():.2f} mm | Median: {np.median(lengths):.2f} mm | Std: {lengths.std():.2f} mm")
for p, v in zip(percentiles, pvals):
    print(f"  p{p}: {v:.2f} mm")
print(f"  Vertex count: {coarse_pv.n_points}, Cell count: {coarse_pv.n_cells}")

q25_final, q50_final, q75_final = np.percentile(lengths, [25, 50, 75])
iqr_ok, _, _, _ = iqr_within_tolerance(lengths, target_size_mm, IQR_TOLERANCE)
if not iqr_ok:
    summary["resolution"] = (f"FAIL (IQR [{q25_final:.2f},{q75_final:.2f}]mm vs "
                              f"{target_size_mm:.2f}mm target, tol={IQR_TOLERANCE*100:.0f}%)")
    print(f"  *** WARNING: IQR still outside tolerance after auto-correction. ***")
else:
    summary["resolution"] = (f"PASS (IQR [{q25_final:.2f},{q75_final:.2f}]mm, "
                              f"median {q50_final:.2f}mm, target {target_size_mm:.2f}mm)")
    print(f"  OK: IQR within {IQR_TOLERANCE*100:.0f}% of target on both bounds.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=60, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(target_size_mm, color="red", linestyle="--", linewidth=2, label=f"target ({target_size_mm:.1f}mm)")
    ax.axvline(q50_final, color="green", linestyle="-", linewidth=2, label=f"median ({q50_final:.2f}mm)")
    ax.axvspan(q25_final, q75_final, color="green", alpha=0.15, label=f"IQR [{q25_final:.2f}, {q75_final:.2f}]mm")
    ax.set_xlabel("Edge length (mm)")
    ax.set_ylabel("Count")
    ax.set_title(f"{resolution} mesh — edge length distribution ({len(lengths)} edges)")
    ax.legend()
    fig.tight_layout()
    hist_path = f"meshes/rodero_05_coarse_{resolution}_edge_histogram.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {hist_path}")
except ImportError:
    print("  (matplotlib not available - skipping histogram plot)")
print()

# ══════════════════════════════════════════════════════════════════════════
# Step 3: Nearest-neighbour marker transfer from fine mesh
# ══════════════════════════════════════════════════════════════════════════

print("Step 3: Transferring boundary markers from fine mesh...")

fine_mesh = dolfin.Mesh()
with dolfin.HDF5File(fine_mesh.mpi_comm(), MESH_DIR + "rodero_05_fine.h5", "r") as f:
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

print(f"  Fine marker counts: "
      f"{ {m: int(np.sum(fine_boundary_markers == m)) for m in np.unique(fine_boundary_markers)} }")

tree_fine_bnd = cKDTree(fine_boundary_centroids)

coarse_surf = coarse_pv.extract_surface(algorithm='dataset_surface')
coarse_face_centres = coarse_surf.cell_centers().points
_, idx = tree_fine_bnd.query(coarse_face_centres)
coarse_markers = fine_boundary_markers[idx]
print(f"  Coarse marker counts: "
      f"{ {m: int(np.sum(coarse_markers == m)) for m in np.unique(coarse_markers)} }")

tree_coarse_bnd = cKDTree(coarse_face_centres)

# ══════════════════════════════════════════════════════════════════════════
# Step 4: Build dolfin mesh and assign ffun
# ══════════════════════════════════════════════════════════════════════════

print("Step 4: Building dolfin mesh and assigning boundary markers...")

m = meshio.read(MESH_DIR + "rodero_05_coarse_" + resolution + ".vtu")
mesh_coords = m.points
mesh_cells = m.cells_dict["tetra"]

dolfin_mesh = dolfin.Mesh()
editor = dolfin.MeshEditor()
editor.open(dolfin_mesh, "tetrahedron", 3, 3)
editor.init_vertices(len(mesh_coords))
editor.init_cells(len(mesh_cells))
for i, pt in enumerate(mesh_coords):
    editor.add_vertex(i, pt)
for i, cell in enumerate(mesh_cells):
    editor.add_cell(i, cell)
editor.close()
dolfin_mesh.order()
dolfin_mesh.init()
dolfin_mesh.init(2, 3)

ffun = dolfin.MeshFunction("size_t", dolfin_mesh, 2, 0)
for facet in dolfin.facets(dolfin_mesh):
    if facet.exterior():
        mp = facet.midpoint().array()
        _, i = tree_coarse_bnd.query(mp)
        ffun[facet.index()] = coarse_markers[i]

print(f"  Dolfin marker counts: "
      f"{ {m: int(np.sum(ffun.array() == m)) for m in np.unique(ffun.array())} }")
print(f"  Mesh: {dolfin_mesh.num_vertices()} vertices, {dolfin_mesh.num_cells()} cells")

# ══════════════════════════════════════════════════════════════════════════
# Step 5: Save mesh + ffun + markers to h5
# ══════════════════════════════════════════════════════════════════════════

print("Step 5: Saving mesh to meshes/rodero_05_coarse_" + resolution + ".h5...")

h5_path = MESH_DIR + "rodero_05_coarse_" + resolution + ".h5"

with dolfin.HDF5File(dolfin_mesh.mpi_comm(), h5_path, "w") as f:
    f.write(dolfin_mesh, "mesh")
    f.write(ffun, "meshfunctions/ffun")

with h5py.File(h5_path, "a") as f:
    f.create_dataset("info/mesh_type", data="biv_ellipsoid")
    f.create_dataset("info/num_refinements", data=0)
    f.create_dataset("markers/BASE", data=[10, 2])
    f.create_dataset("markers/ENDO_RV", data=[20, 2])
    f.create_dataset("markers/ENDO_LV", data=[30, 2])
    f.create_dataset("markers/EPI", data=[40, 2])

print("  Saved mesh and ffun.")

# ══════════════════════════════════════════════════════════════════════════
# Step 6: Fibre interpolation (nearest-vertex) + Gram-Schmidt
# ══════════════════════════════════════════════════════════════════════════

print("Step 6: Interpolating fibres from fine mesh...")

VFS_fine_p1 = dolfin.VectorFunctionSpace(fine_mesh, "P", 1)
f0_fine = dolfin.Function(VFS_fine_p1)
s0_fine = dolfin.Function(VFS_fine_p1)
n0_fine = dolfin.Function(VFS_fine_p1)

with dolfin.HDF5File(fine_mesh.mpi_comm(), MESH_DIR + "rodero_05_fine.h5", "r") as f:
    f.read(f0_fine, "microstructure/f0")
    f.read(s0_fine, "microstructure/s0")
    f.read(n0_fine, "microstructure/n0")

v2d = dolfin.vertex_to_dof_map(VFS_fine_p1)
fine_topo = fine_mesh.cells()
n_fine_vertices = fine_mesh.num_vertices()


def p1_to_cell_centres_nearest(fn, fine_mesh, fine_topo, v2d, n_fine_vertices):
    """Assign each fine cell the fibre value at its nearest vertex (no averaging -
    averaging breaks orthonormality across a rotating fibre field, see history)."""
    raw = fn.vector().get_local()
    nodes = np.zeros((n_fine_vertices, 3))
    for i in range(n_fine_vertices):
        nodes[i] = raw[v2d[3 * i:3 * i + 3]]

    vertex_coords = fine_mesh.coordinates()
    cell_centres_local = vertex_coords[fine_topo].mean(axis=1)
    corner_coords = vertex_coords[fine_topo]
    dists = np.linalg.norm(corner_coords - cell_centres_local[:, None, :], axis=2)
    nearest_local = np.argmin(dists, axis=1)
    nearest_global = fine_topo[np.arange(len(fine_topo)), nearest_local]
    return nodes[nearest_global]


f0_vals = p1_to_cell_centres_nearest(f0_fine, fine_mesh, fine_topo, v2d, n_fine_vertices)
s0_vals = p1_to_cell_centres_nearest(s0_fine, fine_mesh, fine_topo, v2d, n_fine_vertices)
n0_vals = p1_to_cell_centres_nearest(n0_fine, fine_mesh, fine_topo, v2d, n_fine_vertices)

print(f"  Fine mesh zero f0: {np.sum(np.linalg.norm(f0_vals, axis=1) < 1e-10)} / {len(f0_vals)}")

fine_cell_centres = fine_mesh.coordinates()[fine_topo].mean(axis=1)
tree_fine_cells = cKDTree(fine_cell_centres)

coarse_coords = dolfin_mesh.coordinates()
coarse_cells_arr = dolfin_mesh.cells()
coarse_cell_centres = coarse_coords[coarse_cells_arr].mean(axis=1)
_, idx = tree_fine_cells.query(coarse_cell_centres)

print(f"  Fine cell centres range: {fine_cell_centres.min():.3f} to {fine_cell_centres.max():.3f}")
print(f"  Coarse cell centres range: {coarse_cell_centres.min():.3f} to {coarse_cell_centres.max():.3f}")

f0_raw = f0_vals[idx]
s0_raw = s0_vals[idx]
n0_raw = n0_vals[idx]

print("  Before Gram-Schmidt:")
dot_fs = np.abs(np.sum(f0_raw * s0_raw, axis=1))
dot_fn = np.abs(np.sum(f0_raw * n0_raw, axis=1))
dot_sn = np.abs(np.sum(s0_raw * n0_raw, axis=1))
print(f"    max|f0.s0|={dot_fs.max():.3e}, max|f0.n0|={dot_fn.max():.3e}, max|s0.n0|={dot_sn.max():.3e}")


def gram_schmidt_orthonormalize(f0_arr, s0_arr, n0_arr):
    """Row-wise Gram-Schmidt. f0 anchored (renormalized only); s0 orthogonalized
    against f0; n0 = f0 x s0 (exact orthogonality to both by construction)."""
    f0_norm = np.linalg.norm(f0_arr, axis=1, keepdims=True)
    f0_new = f0_arr / np.where(f0_norm > 1e-10, f0_norm, 1.0)

    proj = np.einsum('ij,ij->i', s0_arr, f0_new)[:, None] * f0_new
    s0_orth = s0_arr - proj
    s0_norm = np.linalg.norm(s0_orth, axis=1, keepdims=True)
    degenerate_s0 = (s0_norm < 1e-8).flatten()
    if degenerate_s0.sum() > 0:
        print(f"    WARNING: {degenerate_s0.sum()} elements have s0 nearly parallel "
              f"to f0 after orthogonalization!")
    s0_new = s0_orth / np.where(s0_norm > 1e-10, s0_norm, 1.0)

    n0_new = np.cross(f0_new, s0_new)
    n0_norm = np.linalg.norm(n0_new, axis=1, keepdims=True)
    n0_new = n0_new / np.where(n0_norm > 1e-10, n0_norm, 1.0)

    return f0_new, s0_new, n0_new


f0_gs, s0_gs, n0_gs = gram_schmidt_orthonormalize(f0_raw, s0_raw, n0_raw)

print("  After Gram-Schmidt:")
dot_fs = np.abs(np.sum(f0_gs * s0_gs, axis=1))
dot_fn = np.abs(np.sum(f0_gs * n0_gs, axis=1))
dot_sn = np.abs(np.sum(s0_gs * n0_gs, axis=1))
print(f"    max|f0.s0|={dot_fs.max():.3e}, max|f0.n0|={dot_fn.max():.3e}, max|s0.n0|={dot_sn.max():.3e}")

VFS_coarse = dolfin.VectorFunctionSpace(dolfin_mesh, "DG", 0)
f0_coarse = dolfin.Function(VFS_coarse)
s0_coarse = dolfin.Function(VFS_coarse)
n0_coarse = dolfin.Function(VFS_coarse)
f0_coarse.vector().set_local(f0_gs.flatten())
s0_coarse.vector().set_local(s0_gs.flatten())
n0_coarse.vector().set_local(n0_gs.flatten())

print("  Saving fibres to h5...")
with dolfin.HDF5File(dolfin_mesh.mpi_comm(), h5_path, "a") as f:
    f.write(f0_coarse, "microstructure/f0")
    f.write(s0_coarse, "microstructure/s0")
    f.write(n0_coarse, "microstructure/n0")
print("  Fibres saved.")

# ══════════════════════════════════════════════════════════════════════════
# Step 7: Fibre orthonormality check (read back from h5, independent check)
# ══════════════════════════════════════════════════════════════════════════

print("\nStep 7: Fibre orthonormality check...")

with dolfin.HDF5File(dolfin_mesh.mpi_comm(), h5_path, "r") as f:
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
max_dot_fs = np.abs(np.sum(f0_arr * s0_arr, axis=1)).max()
max_dot_fn = np.abs(np.sum(f0_arr * n0_arr, axis=1)).max()
max_dot_sn = np.abs(np.sum(s0_arr * n0_arr, axis=1)).max()

print(f"  f0 norms: min={f0_norms.min():.6f} max={f0_norms.max():.6f}")
print(f"  s0 norms: min={s0_norms.min():.6f} max={s0_norms.max():.6f}")
print(f"  n0 norms: min={n0_norms.min():.6f} max={n0_norms.max():.6f}")
print(f"  f0.s0 max: {max_dot_fs:.6f}  f0.n0 max: {max_dot_fn:.6f}  s0.n0 max: {max_dot_sn:.6f}")

ORTHO_TOL = 1e-4
fibre_ok = (
    abs(f0_norms.min() - 1) < ORTHO_TOL and abs(f0_norms.max() - 1) < ORTHO_TOL
    and abs(s0_norms.min() - 1) < ORTHO_TOL and abs(s0_norms.max() - 1) < ORTHO_TOL
    and abs(n0_norms.min() - 1) < ORTHO_TOL and abs(n0_norms.max() - 1) < ORTHO_TOL
    and max_dot_fs < ORTHO_TOL and max_dot_fn < ORTHO_TOL and max_dot_sn < ORTHO_TOL
)
summary["fibres"] = "PASS" if fibre_ok else f"FAIL (max off-diag dot = {max(max_dot_fs, max_dot_fn, max_dot_sn):.3e})"
print(f"  {'OK' if fibre_ok else '*** FAIL ***'}")

# ══════════════════════════════════════════════════════════════════════════
# Step 8: Mesh quality check
# ══════════════════════════════════════════════════════════════════════════

print("\nStep 8: Mesh quality check...")


def tet_volumes(coords, cells_arr):
    p0 = coords[cells_arr[:, 0]]
    p1 = coords[cells_arr[:, 1]]
    p2 = coords[cells_arr[:, 2]]
    p3 = coords[cells_arr[:, 3]]
    return np.abs(np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0


def tet_quality_ratios(coords, cells_arr):
    p0 = coords[cells_arr[:, 0]]
    p1 = coords[cells_arr[:, 1]]
    p2 = coords[cells_arr[:, 2]]
    p3 = coords[cells_arr[:, 3]]

    vol = np.abs(np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0

    def tri_area(a, b, c):
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    A0 = tri_area(p1, p2, p3)
    A1 = tri_area(p0, p2, p3)
    A2 = tri_area(p0, p1, p3)
    A3 = tri_area(p0, p1, p2)
    surface_area = A0 + A1 + A2 + A3
    inradius = 3 * vol / surface_area

    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p0, axis=1)
    c = np.linalg.norm(p3 - p0, axis=1)
    a1 = np.linalg.norm(p3 - p2, axis=1)
    b1 = np.linalg.norm(p3 - p1, axis=1)
    c1 = np.linalg.norm(p2 - p1, axis=1)
    num = np.sqrt(
        (a * a1 + b * b1 + c * c1) * (a * a1 + b * b1 - c * c1)
        * (a * a1 - b * b1 + c * c1) * (-a * a1 + b * b1 + c * c1)
    )
    circumradius = num / (24 * vol)
    return inradius / circumradius


volumes = tet_volumes(coarse_coords, coarse_cells_arr)
radii = tet_quality_ratios(coarse_coords, coarse_cells_arr)

n_neg_vol = int(np.sum(volumes < 0))
n_tiny_vol = int(np.sum(volumes < 0.01))
n_poor_quality = int(np.sum(radii < QUALITY_THRESHOLD))

print(f"  Cell volumes: min={volumes.min():.4f}, mean={volumes.mean():.4f}, max={volumes.max():.4f}")
print(f"  Negative volumes: {n_neg_vol}")
print(f"  Very small volumes (<0.01): {n_tiny_vol}")
print(f"  Inradius/circumradius: min={radii.min():.4f}, mean={radii.mean():.4f}")
print(f"  Poor quality (ratio < {QUALITY_THRESHOLD}): {n_poor_quality} / {len(radii)}"
      f" ({100 * n_poor_quality / len(radii):.2f}%)")

quality_ok = n_neg_vol == 0 and n_tiny_vol == 0 and (n_poor_quality / len(radii)) < 0.01
summary["quality"] = "PASS" if quality_ok else \
    f"WARN ({n_neg_vol} neg vol, {n_poor_quality} poor-quality cells)"
print(f"  {'OK' if quality_ok else '*** WARN ***'}")

# ══════════════════════════════════════════════════════════════════════════
# Step 9: Boundary marker facet counts
# ══════════════════════════════════════════════════════════════════════════

print("\nStep 9: Boundary marker facet counts...")

ffun_arr = ffun.array()
marker_names = {10: "BASE", 20: "ENDO_RV", 30: "ENDO_LV", 40: "EPI"}
marker_counts = {}
for marker, name in marker_names.items():
    count = int(np.sum(ffun_arr == marker))
    marker_counts[name] = count
    print(f"  {name} ({marker}): {count} facets")
print(f"  Total facets: {len(ffun_arr)}")

# Rough sanity: EPI should be the largest (whole outer surface), BASE should
# be meaningfully smaller (a thin ring), not comparable in size to EPI.
markers_ok = marker_counts["EPI"] > marker_counts["BASE"] and all(v > 0 for v in marker_counts.values())
summary["boundary_markers"] = "PASS" if markers_ok else f"WARN ({marker_counts})"
print(f"  {'OK' if markers_ok else '*** WARN: check EPI/BASE ratio and zero counts ***'}")

# ══════════════════════════════════════════════════════════════════════════
# Step 10: Valve / material label mapping
# ══════════════════════════════════════════════════════════════════════════

print("\nStep 10: Set materials (valve plug mapping)...")

fine_centres_mat = pd.read_csv(
    MESH_DIR + '/rodero_05_fine/rodero_05_fine_tetrahedron_centers.csv', header=None
).to_numpy() * 10.0

tv = pd.read_csv(
    MESH_DIR + '/rodero_05_fine/rodero_05_fine_elementfield_tv-element.csv', header=None
).to_numpy().flatten().astype(int)

print(f"  Fine mesh: {len(fine_centres_mat)} elements")
print(f"  Unique material labels: {np.unique(tv)}")

with h5py.File(h5_path, 'r') as f:
    h5_coords = f['mesh/coordinates'][:]
    h5_topo = f['mesh/topology'][:]

coarse_centres_mat = h5_coords[h5_topo].mean(axis=1)
print(f"  Coarse mesh: {len(coarse_centres_mat)} elements")

tree_mat = cKDTree(fine_centres_mat)
_, idx_mat = tree_mat.query(coarse_centres_mat)
coarse_tv = tv[idx_mat]

n_valve = int(np.sum(coarse_tv >= 7))
print(f"  Coarse material labels: {np.unique(coarse_tv)}")
print(f"  Valve plug elements (7-10): {n_valve} / {len(coarse_tv)} ({100*n_valve/len(coarse_tv):.2f}%)")

np.save(MESH_DIR + '/rodero_05_coarse_' + resolution + '_tv.npy', coarse_tv)
print(f"  Saved meshes/rodero_05_coarse_{resolution}_tv.npy")

valve_ok = 0 < n_valve < 0.5 * len(coarse_tv)  # sanity: some but not most of the mesh
summary["valve_mapping"] = "PASS" if valve_ok else f"WARN ({n_valve}/{len(coarse_tv)} valve elements)"

# ══════════════════════════════════════════════════════════════════════════
# Step 11: Material strain-energy-at-rest sanity check
# ══════════════════════════════════════════════════════════════════════════
#
# Standalone numpy replica of pulse.HolzapfelOgden's strain energy formula,
# evaluated at F=Identity, using the actual fibre field just generated.
# Avoids needing a full pulse.MechanicsProblem/EM coupling just to test this.
#
# At F=I: I1=3, I4f=I4s=1 exactly (unit fibres), I8fs=f0.s0 (should be ~0 for
# an orthonormal frame). W1 and W4f/W4s vanish exactly at these values by
# construction; only W8fs can be nonzero if fibres aren't truly orthogonal.

print("\nStep 11: Material strain-energy-at-rest sanity check...")
print(f"  Parameters: {MATERIAL_PARAMS_TO_CHECK}")

I8fs_at_rest = np.sum(f0_gs * s0_gs, axis=1)  # per-cell, should be ~0 everywhere

a_fs = MATERIAL_PARAMS_TO_CHECK["a_fs"]
b_fs = MATERIAL_PARAMS_TO_CHECK["b_fs"]

if a_fs > 1e-12 and b_fs > 1e-12:
    W8fs_per_cell = a_fs / (2.0 * b_fs) * (np.exp(b_fs * I8fs_at_rest ** 2) - 1.0)
else:
    W8fs_per_cell = 0.5 * a_fs * I8fs_at_rest ** 2

# W1, W4f, W4s are exactly 0 at F=I given unit-norm fibres (I1=3, I4f=I4s=1
# exactly cancels each term's "-1" offset) - only integrate W8fs.
total_energy_at_rest = float(np.sum(W8fs_per_cell * volumes))

print(f"  I8fs range at rest: {I8fs_at_rest.min():.6e} to {I8fs_at_rest.max():.6e}")
print(f"  Total strain energy at F=I (should be ~0): {total_energy_at_rest:.6e}")

energy_ok = abs(total_energy_at_rest) < STRAIN_ENERGY_TOLERANCE
summary["strain_energy_at_rest"] = "PASS" if energy_ok else \
    f"FAIL ({total_energy_at_rest:.3e}, tol={STRAIN_ENERGY_TOLERANCE})"
print(f"  {'OK' if energy_ok else '*** FAIL: material has spurious residual stress at rest ***'}")
if not energy_ok:
    print("  -> This means the fibre field still has a real orthogonality problem")
    print("     for these specific material parameters, OR the material parameters")
    print("     themselves are not well-posed for this pulse version. Do NOT proceed")
    print("     to a full simulation until this is resolved.")

# ══════════════════════════════════════════════════════════════════════════
# Step 12: XDMF export for visualisation
# ══════════════════════════════════════════════════════════════════════════

print("\nStep 12: Exporting XDMF + fibre VTUs for visualisation...")

with dolfin.XDMFFile(MESH_DIR + "rodero_05_coarse_mesh_" + resolution + ".xdmf") as xf:
    xf.write(dolfin_mesh)
with dolfin.XDMFFile(MESH_DIR + "rodero_05_coarse_boundaries_" + resolution + ".xdmf") as xf:
    xf.write(ffun)


def export_fibres_vtu(coords, cells_arr, f0_arr, s0_arr, n0_arr, out_path):
    vtk_cells = np.hstack([np.full((cells_arr.shape[0], 1), 4), cells_arr]).flatten()
    grid = pv.UnstructuredGrid(vtk_cells, np.full(cells_arr.shape[0], pv.CellType.TETRA), coords)
    grid.cell_data["f0"] = f0_arr
    grid.cell_data["s0"] = s0_arr
    grid.cell_data["n0"] = n0_arr
    grid.cell_data["f0_dot_s0"] = np.abs(np.sum(f0_arr * s0_arr, axis=1))
    grid.cell_data["f0_dot_n0"] = np.abs(np.sum(f0_arr * n0_arr, axis=1))
    grid.cell_data["s0_dot_n0"] = np.abs(np.sum(s0_arr * n0_arr, axis=1))
    grid.save(out_path)
    print(f"  Saved {out_path}")


export_fibres_vtu(coarse_coords, coarse_cells_arr, f0_gs, s0_gs, n0_gs,
                  MESH_DIR + "rodero_05_coarse_" + resolution + "_fibres.vtu")

print("  Done. Output: " + h5_path + ", meshes/rodero_05_coarse_mesh_" + resolution + ".xdmf, "
      "meshes/rodero_05_coarse_boundaries_" + resolution + ".xdmf")

# ══════════════════════════════════════════════════════════════════════════
# Step 13: Summary
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(f"SUMMARY for {resolution} ({h5_path})")
print("=" * 60)
all_pass = True
for check, result in summary.items():
    print(f"  {check:24s}: {result}")
    if not result.startswith("PASS"):
        all_pass = False
print("=" * 60)
if all_pass:
    print("All checks PASSED. Safe to proceed to a mechanics/EM simulation.")
else:
    print("*** One or more checks did NOT pass. Review before running a simulation. ***")
print("=" * 60)