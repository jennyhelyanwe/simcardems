"""
Convert raw mesh arrays into a cardiac_geometries-compatible .h5 file
that simcardems's BiVentricularGeometry (via Geometry.from_file) can read.

Using node_coords/element_connectivity from xyz.csv/tetra.csv (mesh-generator
numbering), paired with ELEMENT-based fibre/sheet/normal fields — these are
defined per-tetrahedron rather than per-node, so they should align with
element_connectivity's row order directly rather than needing node-ID matching.

Marker tags below match BiVentricularGeometry.default_markers() placeholders
(BASE=10, ENDO_RV=20, ENDO_LV=30, EPI=40) — confirm/replace these against
actual face labels.
"""

from pathlib import Path

import dolfin
import numpy as np
import pandas as pd
import pyvista as pv
from cardiac_geometries.geometry import MeshTypes


# --- 1. Build the mesh -------------------------------------------------

def build_mesh(node_coords: np.ndarray, element_connectivity: np.ndarray) -> dolfin.Mesh:
    mesh = dolfin.Mesh()
    editor = dolfin.MeshEditor()
    editor.open(mesh, "tetrahedron", 3, 3)
    editor.init_vertices(node_coords.shape[0])
    editor.init_cells(element_connectivity.shape[0])

    for i, coord in enumerate(node_coords):
        editor.add_vertex(i, coord)

    for i, conn in enumerate(element_connectivity):
        editor.add_cell(i, conn)

    editor.close()
    mesh.init()
    return mesh


# --- 2. Build the facet function (ffun) --------------------------------

def build_ffun(
    mesh: dolfin.Mesh,
    surface_triangles: np.ndarray,
    face_labels: np.ndarray,
    markers: dict,
    label_map: dict,
) -> dolfin.MeshFunction:
    ffun = dolfin.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    mesh.init(2, 0)

    facet_lookup = {}
    for facet in dolfin.facets(mesh):
        verts = tuple(sorted(facet.entities(0)))
        facet_lookup[verts] = facet.index()

    for tri, raw_label in zip(surface_triangles, face_labels):
        verts = tuple(sorted(int(v) for v in tri))
        facet_idx = facet_lookup.get(verts)
        if facet_idx is None:
            raise ValueError(
                f"Triangle {tri} not found among mesh facets — "
                "check node numbering / 0-indexing consistency with build_mesh"
            )
        marker_key = label_map[raw_label]
        tag = markers[marker_key][0]
        ffun[facet_idx] = tag

    return ffun


# --- 3. Build fiber/sheet/normal microstructure (NODE-based, P1) ----
def build_microstructure(
    mesh: dolfin.Mesh,
    fiber_vectors: np.ndarray,
    sheet_vectors: np.ndarray,
    normal_vectors: np.ndarray,
):
    """
    fiber_vectors/sheet_vectors/normal_vectors: (n_nodes, 3) float arrays,
    defined per node (vertex) — uses "P_1" vector space, matching the
    convention cardiac_geometries itself requires for per-vertex fiber data
    (see fibers/_biv_ellipsoid.py's assert fiber_space == "P_1" for the
    vertex-mapped path).

    NOTE: dolfin's P_1 VectorFunctionSpace dof ordering is NOT guaranteed
    to be a simple [v0x,v0y,v0z, v1x,v1y,v1z, ...] flatten matching vertex
    index order directly — it depends on dof-to-vertex mapping, which can
    differ under MPI/parallel reordering. For a single-process conversion
    this is normally fine via dolfin.vertex_to_dof_map, used below to be
    safe rather than assuming raw flatten order.
    """
    V = dolfin.VectorFunctionSpace(mesh, "P", 1)

    v2d = dolfin.vertex_to_dof_map(V)
    # v2d has length n_vertices * 3; v2d[3*i:3*i+3] gives the dof indices
    # for vertex i's (x,y,z) components, in mesh-vertex order.

    def make_function(vectors: np.ndarray) -> dolfin.Function:
        f = dolfin.Function(V)
        vec = f.vector().get_local()
        n_vertices = mesh.num_vertices()
        for i in range(n_vertices):
            dofs = v2d[3 * i : 3 * i + 3]
            vec[dofs] = vectors[i]
        f.vector().set_local(vec)
        f.vector().apply("insert")
        return f

    f0 = make_function(fiber_vectors)
    s0 = make_function(sheet_vectors)
    n0 = make_function(normal_vectors)

    return f0, s0, n0

# --- 4. Write everything to the cardiac_geometries .h5 format -----------

def save_biv_geometry(
    output_path: str,
    mesh: dolfin.Mesh,
    ffun: dolfin.MeshFunction,
    f0: dolfin.Function,
    s0: dolfin.Function,
    n0: dolfin.Function,
    markers: dict,
    info: dict,
):
    path = Path(output_path).with_suffix(".h5")
    if path.is_file():
        path.unlink()

    comm = mesh.mpi_comm()

    with dolfin.HDF5File(comm, path.as_posix(), "w") as h5file:
        h5file.write(mesh, "/mesh")
        h5file.write(ffun, "/meshfunctions/ffun")
        h5file.write(f0, "/microstructure/f0")
        h5file.write(s0, "/microstructure/s0")
        h5file.write(n0, "/microstructure/n0")

    import h5py

    with h5py.File(path, "a") as h5file:
        markers_group = h5file.create_group("/markers")
        for k, v in markers.items():
            markers_group.create_dataset(k, data=v)

        info_group = h5file.create_group("/info")
        for k, v in info.items():
            info_group.create_dataset(k, data=v)

    print(f"Wrote BiV geometry to {path}")


# --- 5. VTU export for visual sanity check (element/cell data) ----------

def export_raw_microstructure_vtu_elementwise(node_coords, element_connectivity,
                                               fibre_vectors, sheet_vectors, normal_vectors,
                                               out_path="rodero_05_fine_raw_fibres_element.vtu"):
    """
    Element-based fields go in as CELL data (one vector per cell), not point data.
    """
    n_cells = element_connectivity.shape[0]
    vtk_cells = np.hstack(
        [np.full((n_cells, 1), 4), element_connectivity]
    ).astype(np.int64).flatten()
    grid = pv.UnstructuredGrid(vtk_cells, np.full(n_cells, pv.CellType.TETRA), node_coords)

    grid.cell_data["f0"] = fibre_vectors
    grid.cell_data["s0"] = sheet_vectors
    grid.cell_data["n0"] = normal_vectors

    f0n = np.linalg.norm(fibre_vectors, axis=1)
    s0n = np.linalg.norm(sheet_vectors, axis=1)
    n0n = np.linalg.norm(normal_vectors, axis=1)
    grid.cell_data["f0_norm"] = f0n
    grid.cell_data["s0_norm"] = s0n
    grid.cell_data["n0_norm"] = n0n
    grid.cell_data["f0_dot_s0"] = np.abs(np.sum(fibre_vectors * sheet_vectors, axis=1))
    grid.cell_data["f0_dot_n0"] = np.abs(np.sum(fibre_vectors * normal_vectors, axis=1))
    grid.cell_data["s0_dot_n0"] = np.abs(np.sum(sheet_vectors * normal_vectors, axis=1))

    grid.save(out_path)
    print(f"  Saved {out_path}")
    print(f"  f0 norm range: {f0n.min():.4f} to {f0n.max():.4f}")
    print(f"  s0 norm range: {s0n.min():.4f} to {s0n.max():.4f}")
    print(f"  n0 norm range: {n0n.min():.4f} to {n0n.max():.4f}")
    print(f"  max|f0.s0|: {grid.cell_data['f0_dot_s0'].max():.6f}")
    print(f"  max|f0.n0|: {grid.cell_data['f0_dot_n0'].max():.6f}")
    print(f"  max|s0.n0|: {grid.cell_data['s0_dot_n0'].max():.6f}")


# ── Main pipeline ────────────────────────────────────────────────────────

dir = './rodero_05_fine'
info = {"mesh_type": MeshTypes.biv_ellipsoid.value}  # == 3


import numpy as np

def read_ensi_vector(path, n_nodes, header_lines=4):
    """
    Reads an Alya EnSight Gold per-node vector variable file (ASCII, block format):
    header_lines lines of metadata, then three blocks of n_nodes values each
    (all X, then all Y, then all Z). Node ordering is implicit: row i in each
    block = node (i+1), matching the .geo file's node id block order.

    Returns (n_nodes, 3) array.
    """
    values = np.loadtxt(path, skiprows=header_lines)
    expected = n_nodes * 3
    assert values.shape[0] == expected, (
        f"{path}: expected {expected} values ({n_nodes} nodes x 3), got {values.shape[0]}"
    )

    x = values[0:n_nodes]
    y = values[n_nodes:2*n_nodes]
    z = values[2*n_nodes:3*n_nodes]

    return np.column_stack([x, y, z])


def read_ensi_geo(path):
    """
    Reads an Alya EnSight Gold geometry file (ASCII, 'node id given' / 'element id given'):
    header block -> node id block (n_nodes) -> coordinate blocks (x, y, z, each n_nodes)
    -> element section (type keyword, element id block, connectivity block).

    Returns:
      node_coords: (n_nodes, 3), ordered by node id (row i = node id i+1, matching
                   the ensi vector files' implicit ordering)
      element_connectivity: (n_elements, 4) int array, 0-indexed
    """
    with open(path, 'r') as f:
        lines = [line.rstrip('\n') for line in f]

    idx = 0
    # Header lines: "Problem name:", "Geometry file", "node id given",
    # "element id given", "part", "1", "Volume Mesh", "coordinates"
    while lines[idx].strip().lower() != "coordinates":
        idx += 1
    idx += 1  # move past "coordinates"

    n_nodes = int(lines[idx].strip())
    idx += 1

    node_ids = np.array([int(lines[idx + i].strip()) for i in range(n_nodes)])
    idx += n_nodes

    x = np.array([float(lines[idx + i].strip()) for i in range(n_nodes)])
    idx += n_nodes
    y = np.array([float(lines[idx + i].strip()) for i in range(n_nodes)])
    idx += n_nodes
    z = np.array([float(lines[idx + i].strip()) for i in range(n_nodes)])
    idx += n_nodes

    # Sort by node id defensively, though block format usually implies 1..n order already
    order = np.argsort(node_ids)
    node_coords = np.column_stack([x, y, z])[order]

    # Element section: expect an element-type keyword line (e.g. "tetra4"),
    # then element count, then element id block, then connectivity block
    elem_type_line = lines[idx].strip()
    print(f"  Element type keyword: '{elem_type_line}'")
    idx += 1

    n_elements = int(lines[idx].strip())
    idx += 1

    elem_ids = np.array([int(lines[idx + i].strip()) for i in range(n_elements)])
    idx += n_elements

    conn = []
    for i in range(n_elements):
        parts = lines[idx + i].split()
        conn.append([int(p) for p in parts])
    idx += n_elements
    conn = np.array(conn)

    elem_order = np.argsort(elem_ids)
    element_connectivity = conn[elem_order] - 1  # 1-indexed -> 0-indexed

    return node_coords, element_connectivity


# ── Usage ────────────────────────────────────────────────────────────────

node_coords_cm, element_connectivity = read_ensi_geo(dir + '/ensight/rodero_05_fine.ensi.geo')
node_coords = node_coords_cm * 10  # cm -> mm

n_nodes = node_coords.shape[0]
print(f"Nodes: {n_nodes}, Elements: {element_connectivity.shape[0]}")
print(f"element_connectivity min/max: {element_connectivity.min()}, {element_connectivity.max()}")

fibre_vectors = read_ensi_vector(dir + '/ensight/rodero_05_fine.ensi.fibre', n_nodes)
sheet_vectors = read_ensi_vector(dir + '/ensight/rodero_05_fine.ensi.sheet', n_nodes)
normal_vectors = read_ensi_vector(dir + '/ensight/rodero_05_fine.ensi.normal', n_nodes)

print(f"fibre_vectors shape: {fibre_vectors.shape}")

f0n = np.linalg.norm(fibre_vectors, axis=1)
s0n = np.linalg.norm(sheet_vectors, axis=1)
n0n = np.linalg.norm(normal_vectors, axis=1)
dot_fs = np.abs(np.sum(fibre_vectors * sheet_vectors, axis=1))
dot_fn = np.abs(np.sum(fibre_vectors * normal_vectors, axis=1))
dot_sn = np.abs(np.sum(sheet_vectors * normal_vectors, axis=1))
print(f"f0 norm range: {f0n.min():.4f} to {f0n.max():.4f}")
print(f"max|f0.s0|: {dot_fs.max():.6f}  max|f0.n0|: {dot_fn.max():.6f}  max|s0.n0|: {dot_sn.max():.6f}")

def export_raw_microstructure_vtu(node_coords, element_connectivity,
                                   fibre_vectors, sheet_vectors, normal_vectors,
                                   out_path="rodero_05_fine_ensi_fibres.vtu"):
    import pyvista as pv
    n_cells = element_connectivity.shape[0]
    vtk_cells = np.hstack(
        [np.full((n_cells, 1), 4), element_connectivity]
    ).astype(np.int64).flatten()
    grid = pv.UnstructuredGrid(vtk_cells, np.full(n_cells, pv.CellType.TETRA), node_coords)

    grid.point_data["f0"] = fibre_vectors
    grid.point_data["s0"] = sheet_vectors
    grid.point_data["n0"] = normal_vectors
    grid.save(out_path)
    print(f"  Saved {out_path}")

export_raw_microstructure_vtu(node_coords, element_connectivity,
                               fibre_vectors, sheet_vectors, normal_vectors)

print("Reading surface triangles and face labels...")
df = pd.read_csv(dir + '/rodero_05_fine_triangles.csv', header=None)
surface_triangles = df.to_numpy()
df = pd.read_csv(dir + '/rodero_05_fine_boundaryelementfield_mechanical-element-boundary-label.csv', header=None)
face_labels = df.to_numpy().flatten()

def compute_normal_from_cross_product(fibre_vectors, sheet_vectors):
    """
    Computes n0 = f0 x s0, normalized. This guarantees n0 is exactly
    orthogonal to both f0 and s0 (up to floating point), rather than
    relying on a separately-generated field that might not be perfectly
    orthonormal with the other two.
    """
    n0 = np.cross(fibre_vectors, sheet_vectors)
    norms = np.linalg.norm(n0, axis=1, keepdims=True)

    zero_mask = (norms < 1e-10).flatten()
    if zero_mask.sum() > 0:
        print(f"  WARNING: {zero_mask.sum()} elements have f0 parallel to s0 "
              f"(zero cross product) — these elements have degenerate microstructure!")

    n0_normalized = np.where(norms > 1e-10, n0 / norms, n0)
    return n0_normalized

label_map = {4: "BASE", 3: "ENDO_LV", 2: "ENDO_RV", 1: "EPI"}
markers = {"BASE": (10, 2), "ENDO_RV": (20, 2), "ENDO_LV": (30, 2), "EPI": (40, 2)}

# Stop here to inspect the VTU in ParaView before trusting the rest of the pipeline.
# Uncomment once it looks correct.

print('Building mesh using node coordinates and element connectivity')
mesh = build_mesh(node_coords, element_connectivity)
print('Building facet function using face labels')
ffun = build_ffun(mesh, surface_triangles, face_labels, markers, label_map)
print('Building microstructure (element-based, DG0)')
f0, s0, n0 = build_microstructure(mesh, fibre_vectors, sheet_vectors, normal_vectors)
print('Saving to h5 geometry that simcardems will read')
save_biv_geometry(dir, mesh, ffun, f0, s0, n0, markers, info)

import h5py
with h5py.File("rodero_05_fine.h5", "a") as hf:
    hf["info"].create_dataset("num_refinements", data=0)