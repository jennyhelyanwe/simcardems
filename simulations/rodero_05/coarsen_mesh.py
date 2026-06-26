import dolfin
from cardiac_geometries.geometry import Geometry
import numpy as np

geo = Geometry.from_file("rodero_05_fine.h5")
mesh = geo.mesh
ffun = geo.ffun

# Write to .mesh format with boundary markers
with open("rodero_05_fine.mesh", "w") as f:
    f.write("MeshVersionFormatted 2\nDimension 3\n\n")

    coords = mesh.coordinates()
    f.write(f"Vertices\n{len(coords)}\n")
    for pt in coords:
        f.write(f"{pt[0]} {pt[1]} {pt[2]} 0\n")

    # Triangles with markers
    triangles = []
    for facet in dolfin.facets(mesh):
        if ffun[facet] > 0:
            verts = facet.entities(0)
            triangles.append((verts[0] + 1, verts[1] + 1, verts[2] + 1, int(ffun[facet])))

    f.write(f"\nTriangles\n{len(triangles)}\n")
    for t in triangles:
        f.write(f"{t[0]} {t[1]} {t[2]} {t[3]}\n")

    # Tetrahedra
    cells = mesh.cells()
    f.write(f"\nTetrahedra\n{len(cells)}\n")
    for c in cells:
        f.write(f"{c[0] + 1} {c[1] + 1} {c[2] + 1} {c[3] + 1} 0\n")

    f.write("\nEnd\n")

print("Done")
print("Markers:", set(ffun.array()))

# then use mmg. you may need to run this somewhere else where mmg is avaialble.
# mmg3d_O3 -in rodero_05_fine.mesh -out rodero_05_coarse.mesh -hmax 8.0 -hmin 2.0 -hausd 2.0