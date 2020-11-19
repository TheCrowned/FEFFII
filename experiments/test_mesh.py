from dolfin import *
from mshr import *
import numpy as np
from pylab import show,triplot

set_log_active(True)
set_log_level(1)

def DeformMeshCoords(mesh_coords):    
    x = mesh_coords[:, 0]
    y = mesh_coords[:, 1]

    y=y+0.1*np.sin(2*pi*x)
    
    ret = []
    for i in range(len(x)):
        ret.append(Point(x[i], y[i]))

    mesh_coords = np.array([x, y]).transpose()
    return ret
    
domain_vertices = [Point(0.0, 0.0),
                  Point(1.0, 0),
                  Point(1.0, 1.0),
                  Point(0.0, 1.0)]
                  
domain = Polygon(domain_vertices)
print(type(domain))
mesh = generate_mesh(domain,16)
bmesh = BoundaryMesh(mesh, "exterior", True)
print(bmesh.coordinates())
#new_mesh_coords = DeformMeshCoords(mesh.coordinates())
plot(bmesh)
show()

domain = [Point(p) for p in bmesh.coordinates()]
print((domain))
domain = Polygon(domain) #mesh.coordinates()
print(type(domain))

mesh = generate_mesh(domain,16)
plot(domain)
#plot(coords[:,0], coords[:,1], triangles=mesh.cells())

show()

quit()

'''

"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for channel flow (Poisseuille) on the unit square using the
Incremental Pressure Correction Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
								 div(u) = 0
"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt

T = 10.0           # final time
num_steps = 50    # number of time steps
dt = T / num_steps # time step size
mu = 1             # kinematic viscosity
rho = 1            # density

domain = Rectangle(Point(0., 0.), Point(1., 1.))  
         #Circle(dolfin.Point(0.5, 0.0), 0.1) - \
         #Rectangle(dolfin.Point(0.0, 0.9), dolfin.Point(0.4, 1.0)) 
         
print(domain)
domain = Polygon(domain)
mesh = generate_mesh(domain, 4, "cgal")
print(mesh)
quit()




def DeformMeshCoords(mesh_coords):    
    x = mesh_coords[:, 0]
    y = mesh_coords[:, 1]

    y=y+0.1*np.sin(2*pi*x)
    
    ret = []
    for i in range(len(x)):
        ret.append(Point(x[i], y[i]))

    mesh_coords = np.array([x, y]).transpose()
    return ret

#print(mesh.coordinates())
mesh=DeformMeshCoords(mesh.coordinates())
#print(mesh)


print(new_mesh)

new_mesh = generate_mesh(new_mesh, 4, "cgal")

#plot(new_mesh)

'''
