# feffi module lives one level up current dir
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


from feffi import *
import fenics
from fenics import SubDomain, dot, nabla_grad, grad, div, dx
import matplotlib.pyplot as plt

parameters.define_parameters({
    'config_file' : 'feffi/config/lid-driven-cavity.yml',
    'domain': 'fjord',
    'shelf_size_x':0,
    'shelf_size_y':0,
    #'mesh_resolution': 10
})
#parameters.parse_commandline_args()

mesh = mesh.create_mesh()

f_spaces = functions.define_function_spaces(mesh)
f = functions.define_functions(f_spaces)

domain = boundaries.Domain(mesh, f_spaces)

simul = simulation.Simulation(f, domain.BCs)
for i in range(3):
    simul.timestep()

#plot.plot_solutions(f)

bmesh = fenics.BoundaryMesh(mesh, "exterior")
for x in bmesh.coordinates():
    x[1] *= 0.9

fenics.ALE.move(mesh, bmesh)

simul.timestep()

plot.plot_single(f['u_'], display=True)