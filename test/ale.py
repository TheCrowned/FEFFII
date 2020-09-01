from fenics import *
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(10,10)
top = CompiledSubDomain('near(x[1], 1)')
bottom = CompiledSubDomain('near(x[1], 0)')
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
top.mark(boundaries, 1)
bottom.mark(boundaries, 2)
plot(mesh)
plt.show()

#File('boundaries.pvd') << boundaries

bmesh = BoundaryMesh(mesh, "exterior")
for x in bmesh.coordinates():
	x[1] /= 2

ALE.move(mesh, bmesh)

File('boundaries.pvd') << boundaries