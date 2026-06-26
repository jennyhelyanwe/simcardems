"""
Convert raw mesh arrays into a cardiac_geometries-compatible .h5 file
that simcardems's BiVentricularGeometry (via Geometry.from_file) can read.

Marker tags below match BiVentricularGeometry.default_markers() placeholders
(BASE=10, ENDO_RV=20, ENDO_LV=30, EPI=40) — confirm/replace these against
actual face labels.
"""

import json
from pathlib import Path

import dolfin
import numpy as np
from cardiac_geometries.geometry import MeshTypes


# --- 1. Build the mesh -------------------------------------------------

def build_mesh(node_coords: np.ndarray, element_connectivity: np.ndarray) -> dolfin.Mesh:
    """
    node_coords: (n_nodes, 3) float array
    element_connectivity: (n_elements, 4) int array, tetrahedral, 0-indexed
    """
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
    """
    surface_triangles: (n_faces, 3) int array, 0-indexed node indices,
                        matching the vertex numbering used in build_mesh
    face_labels: (n_faces,) array of raw label values, mapped through
                 label_map onto simcardems's BASE/ENDO_LV/ENDO_RV/EPI keys
    markers: BiVentricularGeometry.default_markers() dict, e.g.
             {"BASE": (10,2), "ENDO_RV": (20,2), "ENDO_LV": (30,2), "EPI": (40,2)}
    label_map: maps your raw face_labels values to marker dict keys, e.g.
               {0: "BASE", 1: "ENDO_LV", 2: "ENDO_RV", 3: "EPI"}
               -- EDIT to match your biv-me label encoding
    """
    ffun = dolfin.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    mesh.init(2, 0)  # ensure facet-vertex connectivity exists

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


# --- 3. Build fiber/sheet/normal microstructure -------------------------

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
    """
    Two-pass write, matching Geometry.save's own pattern:
      pass 1 — dolfin objects (mesh, ffun, f0, s0, n0) via dolfin.HDF5File
      pass 2 — plain dicts (markers, info) via h5py, AFTER the HDF5File
                from pass 1 is closed (dict_to_h5 reopens the same file
                with h5py in append mode; doing this while dolfin's
                HDF5File handle is still open will corrupt the file).
    """
    path = Path(output_path).with_suffix(".h5")
    if path.is_file():
        path.unlink()

    comm = mesh.mpi_comm()

    # Pass 1: dolfin objects
    with dolfin.HDF5File(comm, path.as_posix(), "w") as h5file:
        h5file.write(mesh, "/mesh")
        h5file.write(ffun, "/meshfunctions/ffun")
        h5file.write(f0, "/microstructure/f0")
        h5file.write(s0, "/microstructure/s0")
        h5file.write(n0, "/microstructure/n0")

    # Pass 2: plain dict data, via h5py append (mirrors dict_to_h5 exactly)
    import h5py

    with h5py.File(path, "a") as h5file:
        markers_group = h5file.create_group("/markers")
        for k, v in markers.items():
            markers_group.create_dataset(k, data=v)

        info_group = h5file.create_group("/info")
        for k, v in info.items():
            info_group.create_dataset(k, data=v)

    print(f"Wrote BiV geometry to {path}")


# --- Example usage --------------------------------------------------
#
# from cardiac_geometries.geometry import MeshTypes
#
# markers = {
#     "BASE": (10, 2),
#     "ENDO_RV": (20, 2),
#     "ENDO_LV": (30, 2),
#     "EPI": (40, 2),
# }
# info = {"mesh_type": MeshTypes.biv_ellipsoid.value}  # == 3
#
# label_map = {0: "BASE", 1: "ENDO_LV", 2: "ENDO_RV", 3: "EPI"}  # EDIT to match biv-me
#
# mesh = build_mesh(node_coords, element_connectivity)
# ffun = build_ffun(mesh, surface_triangles, face_labels, markers, label_map)
# f0, s0, n0 = build_microstructure(mesh, fiber_vectors, sheet_vectors, normal_vectors)
# save_biv_geometry("our_biv_geo", mesh, ffun, f0, s0, n0, markers, info)

# Read in .csv arrays from Alya_pipeline/meta_data/rodero_05_coarse/
import pandas as pd
info = {"mesh_type": MeshTypes.biv_ellipsoid.value}  # == 3
dir = './rodero_05_coarse'
df = pd.read_csv(dir + '/rodero_05_coarse_xyz.csv', header=None)
node_coords = df.to_numpy() * 10 # Because simcardems is in mm-kPa-ms-g units, and Alya meshes are in cm.
df = pd.read_csv(dir + '/rodero_05_coarse_tetra.csv', header=None)
element_connectivity = df.to_numpy().astype(int)
df = pd.read_csv(dir + '/rodero_05_coarse_triangles.csv', header=None)
surface_triangles = df.to_numpy()
df = pd.read_csv(dir + '/rodero_05_coarse_boundaryelementfield_mechanical-element-boundary-label.csv', header=None)
face_labels = df.to_numpy()
df = pd.read_csv(dir + '/rodero_05_coarse_nodefield_fibre.csv', header=None)
fibre_vectors = df.to_numpy()
df = pd.read_csv(dir + '/rodero_05_coarse_nodefield_sheet.csv', header=None)
sheet_vectors = df.to_numpy()
df = pd.read_csv(dir + '/rodero_05_coarse_nodefield_normal.csv', header=None)
nornaml_vectors = df.to_numpy()

import pyvista as pv
def plot_labeled_faces(
    node_coords: np.ndarray,
    surface_triangles: np.ndarray,
    face_labels: np.ndarray,
):
    """
    node_coords: (n_nodes, 3) float array
    surface_triangles: (n_faces, 3) int array, 0-indexed into node_coords
    face_labels: (n_faces,) array of raw label values
    """
    # PyVista wants faces as a flat array: [3, i0, i1, i2, 3, i0, i1, i2, ...]
    n_faces = surface_triangles.shape[0]
    faces = np.hstack(
        [np.full((n_faces, 1), 3), surface_triangles]
    ).astype(np.int64).flatten()

    mesh = pv.PolyData(node_coords, faces)
    mesh.cell_data["label"] = face_labels

    unique_labels = np.unique(face_labels)
    print(f"Unique label values: {unique_labels}")
    print(f"Face count per label:")
    for lbl in unique_labels:
        print(f"  {lbl}: {(face_labels == lbl).sum()} faces")

    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="label",
        cmap="tab10",
        show_edges=False,
        categories=True,
        scalar_bar_args={"title": "face label"},
    )
    plotter.add_axes()
    plotter.show()


def plot_each_label_separately(
    node_coords: np.ndarray,
    surface_triangles: np.ndarray,
    face_labels: np.ndarray,
):
    """
    Same data, but renders each label in its own subplot so you can
    isolate e.g. 'is this patch really the RV endocardium' without the
    others occluding it. Useful once the combined plot narrows down
    which labels are base/epi/endo_lv/endo_rv but you want to confirm
    each one's exact extent.
    """
    unique_labels = np.unique(face_labels)
    n = len(unique_labels)
    cols = min(n, 4)
    rows = -(-n // cols)  # ceil
    face_labels = face_labels.flatten()

    plotter = pv.Plotter(shape=(rows, cols))

    for idx, lbl in enumerate(unique_labels):
        row, col = divmod(idx, cols)
        plotter.subplot(row, col)

        mask = face_labels == lbl
        tris = surface_triangles[mask]
        n_faces = tris.shape[0]
        faces = np.hstack(
            [np.full((n_faces, 1), 3), tris]
        ).astype(np.int64).flatten()

        mesh = pv.PolyData(node_coords, faces)
        plotter.add_mesh(mesh, color="coral")
        plotter.add_text(f"label = {lbl} ({n_faces} faces)", font_size=10)

    plotter.show()

# Visualise surface to check if labels are correct.
# plot_labeled_faces(node_coords, surface_triangles, face_labels)
# plot_each_label_separately(node_coords, surface_triangles, face_labels)

label_map =  {4: "BASE", 3: "ENDO_LV", 2: "ENDO_RV", 1: "EPI"}
markers = {"BASE": (10, 2), "ENDO_RV": (20, 2), "ENDO_LV": (30, 2),"EPI": (40, 2) }


print('Building mesh using node coordinates and element connectivity')
mesh = build_mesh(node_coords, element_connectivity)
print('Building facet function using face labels')
face_labels = face_labels.flatten()
ffun = build_ffun(mesh, surface_triangles, face_labels, markers, label_map)
print('Building microstructure by flattening f, s, n vectors')
f0, s0, n0 = build_microstructure(mesh, fibre_vectors, sheet_vectors, nornaml_vectors)
print('Saving to h5 geometry that simcardems will read')
save_biv_geometry(dir, mesh, ffun, f0, s0, n0, markers, info)

# FOR NOW! Remove EP refinement to speed up debug process. When actually running, do include refinement.
import h5py
with h5py.File("rodero_05_coarse.h5", "a") as hf:
    hf["info"].create_dataset("num_refinements", data=0)

