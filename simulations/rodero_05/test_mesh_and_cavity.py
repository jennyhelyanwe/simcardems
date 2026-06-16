import h5py

# with h5py.File("rodero_05_fine.h5", "r") as hf:
#     def show(name, obj):
#         if isinstance(obj, h5py.Dataset):
#             print(f"DATASET {name}: shape={obj.shape}, dtype={obj.dtype}")
#             for k, v in obj.attrs.items():
#                 print(f"    attr: {k} = {v}")
#         else:
#             print(f"GROUP   {name}")
#     hf.visititems(show)
#     print("--- top-level keys ---")
#     print(list(hf.keys()))

import dolfin

# mesh = dolfin.Mesh()
# with dolfin.HDF5File(mesh.mpi_comm(), "rodero_05_fine.h5", "r") as h5file:
#     h5file.read(mesh, "/mesh", True)
#
# V = dolfin.VectorFunctionSpace(mesh, "P", 1)
# f = dolfin.Function(V)
# with dolfin.HDF5File(mesh.mpi_comm(), "/tmp/test_sig.h5", "w") as h5file:
#     h5file.write(f, "/test")
#
# import h5py
# with h5py.File("/tmp/test_sig.h5", "r") as hf:
#     print(dict(hf["test"].attrs))
#
# import h5py
#
# with h5py.File("rodero_05_fine.h5", "r") as hf:
#     print(dict(hf["microstructure/f0"].attrs))
#     print(dict(hf["microstructure/f0/vector_0"].attrs))

import h5py

# with h5py.File("rodero_05_fine.h5", "a") as hf:
#     hf["info"].create_dataset("num_refinements", data=0)

from cardiac_geometries.geometry import Geometry

geo = Geometry.from_file("rodero_05_fine.h5")
# print(geo)
# print(geo.mesh.num_vertices(), geo.mesh.num_cells())
# print(geo.markers)
# print(geo.info)
# print(geo.f0.function_space())

from simcardems.bivgeometry import BiVentricularGeometry

biv_geo = BiVentricularGeometry.from_geometry(
    geo,
    ep_mesh=geo.mesh,
    ffun_ep=geo.ffun,
)
print(biv_geo)
print(biv_geo.markers)
print(biv_geo.mechanics_mesh.num_vertices())
print(biv_geo.microstructure.f0.function_space())

import dolfin
from simcardems.biv_cavity_cycle_controller import compute_cavity_volume

# geo = whatever BiVentricularGeometry / pulse geometry object you have loaded
V = dolfin.VectorFunctionSpace(biv_geo.mesh, "P", 2)  # match whatever space u normally lives in;
                                                    # doesn't actually matter for u=0
u0 = dolfin.Function(V)
u0.vector()[:] = 0.0

lv_marker = biv_geo.markers["ENDO_LV"][0]
rv_marker = biv_geo.markers["ENDO_RV"][0]

v_lv = compute_cavity_volume(biv_geo, u0, lv_marker)
v_rv = compute_cavity_volume(biv_geo, u0, rv_marker)

print(f"LV volume (reference config): {v_lv}")
print(f"RV volume (reference config): {v_rv}")
