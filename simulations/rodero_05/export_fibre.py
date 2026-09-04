"""
export_fibers_ensight.py — one-time export of reference fiber (f0) and
sheet (s0) directions as EnSight, projected DG0 (Quadrature_3 -> one
vector per cell), alongside a matching lambda snapshot, for direct
spatial comparison in ParaView.
"""

import os
import numpy as np
import dolfin

from postprocess_ensight import write_geo_file, write_vector_element_file, write_case_file

from cardiac_geometries.geometry import Geometry
from simcardems.bivgeometry import BiVentricularGeometry

MESH_DIR = "meshes/"
RESOLUTION = "4mm"
OUTDIR = "results_4mm/fiber_check_ensight"

os.makedirs(OUTDIR, exist_ok=True)

geo = Geometry.from_file(MESH_DIR + "rodero_05_coarse_" + RESOLUTION + ".h5")
biv_geo = BiVentricularGeometry.from_geometry(geo, ep_mesh=geo.mesh, ffun_ep=geo.ffun)

mesh = biv_geo.mechanics_mesh
coords = mesh.coordinates()
topo = mesh.cells()

V_dg0_vec = dolfin.VectorFunctionSpace(mesh, "DG", 0)

f0_dg0 = dolfin.Function(V_dg0_vec)
f0_dg0.assign(dolfin.project(biv_geo.f0, V_dg0_vec))
f0_vals = f0_dg0.vector().get_local().reshape(-1, 3)

s0_dg0 = dolfin.Function(V_dg0_vec)
s0_dg0.assign(dolfin.project(biv_geo.s0, V_dg0_vec))
s0_vals = s0_dg0.vector().get_local().reshape(-1, 3)

geo_file = "fibers.geo"
write_geo_file(os.path.join(OUTDIR, geo_file), coords, topo)

write_vector_element_file(os.path.join(OUTDIR, "f0.0001"), f0_vals, "f0")
write_vector_element_file(os.path.join(OUTDIR, "s0.0001"), s0_vals, "s0")

case_path = os.path.join(OUTDIR, "fibers.case")
write_case_file(
    case_path, geo_file,
    scalar_vars=[], vector_vars=[], times_float=[0.0],
    vector_element_vars=[("f0", "f0.****"), ("s0", "s0.****")],
)
print(f"Fiber and sheet fields written: {case_path}")