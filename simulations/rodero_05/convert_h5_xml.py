# convert_to_xml.py - run once with srun -n 1
import dolfin
import numpy as np
from cardiac_geometries.geometry import Geometry

geo = Geometry.from_file("rodero_05_fine.h5")

# Save mesh and ffun
dolfin.File("rodero_05_mesh.xml") << geo.mesh
dolfin.File("rodero_05_ffun.xml") << geo.ffun

# Save microstructure
dolfin.File("rodero_05_f0.xml") << geo.f0
dolfin.File("rodero_05_s0.xml") << geo.s0
dolfin.File("rodero_05_n0.xml") << geo.n0

# Save markers and info as numpy
np.save("rodero_05_markers.npy", geo.markers)
np.save("rodero_05_info.npy", geo.info)

print("done")