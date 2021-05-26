# feffi module lives one level up current dir
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import feffi
from fenics import *
from mshr import *
import matplotlib.pyplot as plt

feffi.parameters.define_parameters({
    'config_file' : 'feffi/config/lid-driven-cavity.yml',
})
feffi.parameters.parse_commandline_args()

# Unstructured mesh
#points = [Point(0,0), Point(1,0), Point(1,1), Point(0,1)] # square
#points = [Point(0,0), Point(10,0), Point(10,1), Point(0,1)] # long domain
#pentagon = Polygon(points)
#mesh = generate_mesh(pentagon, 16, 'cgal')

# Structured mesh
mesh = UnitSquareMesh(8,8)

feffi.plot.plot_single(mesh, display=True, title="Starting mesh")

f_spaces = feffi.functions.define_function_spaces(mesh)
f = feffi.functions.define_functions(f_spaces)
domain = feffi.boundaries.Domain(mesh, f_spaces)
simul = feffi.simulation.Simulation(f, domain.BCs)

print("Mesh hmin is {}".format(mesh.hmin()))

# Repeatedly run 1 timestep and deform mesh
for x in range(15):
    simul.timestep()

    # 0.02 x-wise displacement
    disp = Expression(("0.02*(left-x[0])", "0"),
                      left=max(mesh.coordinates()[:,0]), degree=2)

    # Move boundary and then mesh
    #boundary = BoundaryMesh(mesh, 'exterior')
    #ALE.move(boundary, disp)
    #ALE.move(mesh, disp)

    # Move mesh according to given expression
    ALE.move(mesh, disp)

    #feffi.plot.plot_single(f['u_'], display=True)
    feffi.plot.plot_single(mesh, display=True,
                           title="Mesh with left x = {}".format(min(mesh.coordinates()[:,0])))
