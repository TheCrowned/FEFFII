# feffii module lives one level up current dir
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from feffii import *
import fenics
import matplotlib.pyplot as plt
import pygmsh

parameters.define_parameters({
    'config_file' : 'feffii/config/rayleigh-benard-convection-noslip.yml',
    'max_iter' : 3000
})
parameters.parse_commandline_args()

domain_size_x = parameters.config['domain_size_x']
domain_size_y = parameters.config['domain_size_y']
mesh_resolution = parameters.config['mesh_resolution']

points =  [(0,0,0), (domain_size_x,0,0), (domain_size_x,domain_size_y,0), (0, domain_size_y,0)]

# Generate mesh
fenics_mesh = False
with pygmsh.geo.Geometry() as geom:
    poly = geom.add_polygon(points, mesh_size=mesh_resolution)
    m = geom.generate_mesh()
    m.write('mesh.xdmf')
    fenics_mesh = mesh.pygmsh2fenics_mesh(m)
    plot.plot_single(fenics_mesh, display=True)

f_spaces = functions.define_function_spaces(fenics_mesh)
f = functions.define_functions(f_spaces)
functions.init_functions(f) # Init functions to closest steady state

domain = boundaries.Domain(
    fenics_mesh,
    f_spaces,
    boundaries = {
      'bottom' : boundaries.Bound_Bottom(fenics_mesh),
      'left' : boundaries.Bound_Left(fenics_mesh),
      'right' : boundaries.Bound_Right(fenics_mesh),
      'top' : boundaries.Bound_Top(fenics_mesh),
    },
    BCs = parameters.config['BCs'])

simul = simulation.Simulation(f, domain)
simul.run()
