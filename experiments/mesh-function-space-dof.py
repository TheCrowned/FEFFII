from dolfin import * 
import numpy as np
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(2, 2)
V = FunctionSpace(mesh, 'CG', 1)
v = Function(V)

dofmap = V.dofmap()
nvertices = mesh.ufl_cell().num_vertices()
print(nvertices)
print(len(v.vector().get_local()))
# Set up a vertex_2_dof list
indices = [dofmap.tabulate_entity_dofs(0, i)[0] for i in range(nvertices)]

vertex_2_dof = dict()
[vertex_2_dof.update(dict(vd for vd in zip(cell.entities(0),
                                        dofmap.cell_dofs(cell.index())[indices])))
                        for cell in cells(mesh)]

# Get the vertex coordinates
X = mesh.coordinates()

# Set the vertex coordinate you want to modify
xcoord, ycoord = 0.5, 0.5

print(type(X))
for i in range(len(X)):
    print('el {}, val {}, dof {}'.format(i, X[i], vertex_2_dof[i]))

# Find the matching vertex (if it exists)
vertex_idx = np.where((X == (xcoord,ycoord)).all(axis = 1))[0] 
if not vertex_idx:
    print('No matching vertex!')
else:
    vertex_idx = vertex_idx[0]
    dof_idx = vertex_2_dof[vertex_idx]
    print(dof_idx)
    print(vertex_idx)
    v.vector()[dof_idx] = 1.

plot(v)
plt.show()
plot(mesh)
plt.show()
