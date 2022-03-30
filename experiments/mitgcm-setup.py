# feffi module lives one level up current dir
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# Force monothread in OpenBLAS, seems to even speed up
# https://stackoverflow.com/questions/52026652/openblas-blas-thread-init-pthread-create-resource-temporarily-unavailable
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import feffi
from fenics import *
import numpy as np
import pygmsh


####################
### Config setup ###
####################

feffi.parameters.define_parameters({
    'config_file' : 'feffi/config/mitgcm-setup.yml',
})
config = feffi.parameters.config

## Config shorthands
domain_size_x = config['domain_size_x']
domain_size_y = config['domain_size_y']
mesh_resolution = config['mesh_resolution']
ice_shelf_bottom_p = config['ice_shelf_bottom_p']
ice_shelf_top_p = config['ice_shelf_top_p']
ice_shelf_slope = config['ice_shelf_slope']
ice_shelf_f = lambda x: ice_shelf_slope*x+ice_shelf_bottom_p[1]

##############################
### Set up geometry + mesh ###
##############################

# FEniCS mesher seems much faster than pyGmsh for fine meshes, but
# pyGmsh allows fine grained control over mesh size at different locations.

# pygmsh handles long boundary refinement badly, so we need to interval the
# ice shelf boundary with many points, see
# https://github.com/nschloe/pygmsh/issues/506
# It is not clear whether the step size has an influence on the mesh quality
shelf_points_x = list(np.arange(ice_shelf_bottom_p[0], ice_shelf_top_p[0], 0.4))
shelf_points_x.reverse()
shelf_points = [(x, ice_shelf_f(x), 0) for x in shelf_points_x[0:-1]] # exclude last (bottom) point to avoid duplicate

sea_top_points_x = list(np.arange(ice_shelf_top_p[0], domain_size_x, 0.4))
sea_top_points_x.reverse()
sea_top_points = [(x, domain_size_y, 0) for x in sea_top_points_x[0:-1]] # exclude last (bottom) point to avoid duplicate

# Points that makes up geometry
points =  [(0,0,0), (domain_size_x,0,0)]                           # bottom
points += [(domain_size_x,domain_size_y,0)]                        # right
points += sea_top_points                                           # sea top
points += [(ice_shelf_top_p[0],domain_size_y,0), ice_shelf_top_p]  # sea top
points += shelf_points                                             # ice shelf
points += [ice_shelf_bottom_p]                                     # left

## IDEA: give points only, and automatically generate a list of points
##       for each segment, as done for ice shelf and sea top, but for 
##       all boundaries.


##############################
### PyGMSH mesh generation ###
##############################

fenics_mesh = False
with pygmsh.geo.Geometry() as geom:
    poly = geom.add_polygon(points, mesh_size=mesh_resolution)

    # Mesh refinement
    '''ice_shelf_lines = [poly.curve_loop.curves[i] for i in range(len(points)-len(shelf_points)-3, len(points)-1)]
    field = geom.add_boundary_layer(
        edges_list=ice_shelf_lines,
        lcmin=0.006,
        lcmax=mesh_resolution, #0.15,
        distmin=0.009, # distance up until which mesh size will be lcmin
        distmax=0.15, # distance starting at which mesh size will be lcmax
    )
    #geom.set_background_mesh([field], operator="Min")'''

    # Generate mesh
    mesh = geom.generate_mesh()
    mesh.write('mesh.xdmf')
    fenics_mesh = feffi.mesh.pygmsh2fenics_mesh(mesh)
    #fenics_mesh = UnitSquareMesh(50,50)
    feffi.plot.plot_single(fenics_mesh, display=True)

# Ice shelf boundary
class Bound_Ice_Shelf(SubDomain):
    def inside(self, x, on_boundary):
        return (((0 < x[0] <= ice_shelf_top_p[0] and ice_shelf_bottom_p[1] <= x[1] <= domain_size_y)
                 or (near(x[0], ice_shelf_top_p[0]) and ice_shelf_top_p[1] <= x[1] <= domain_size_y))
            and on_boundary)


######################
## Simulation setup ##
######################

f_spaces = feffi.functions.define_function_spaces(fenics_mesh)
f = feffi.functions.define_functions(f_spaces)
#feffi.functions.init_functions(f)

# Initial conditions for T,S that mimick Ryder glacier
tcd = -0.8; # Thermocline depth
Tmin, Tmax = -1.6, 0.4; # minimum/maximum Temperature
Tc = (Tmax + Tmin) / 2; # middle temperature
Trange = Tmax - Tmin; # temperature range
T_init = Expression('Tc - Trange / 2 * -tanh(pi * 1000*(-x[1] - tcd) / 200)', degree=1, Tc=Tc, Trange=Trange, tcd=tcd); # calculate profile
f['T_n'].assign(interpolate(T_init, f['T_n'].ufl_function_space()))
#feffi.plot.plot_single(f['T_n'], title='Temperature', display=True)

Sc = 34.5;
Srange = -1;
S_init = Expression('Sc + Srange / 2 * -tanh(pi * 1000*(-x[1] - tcd) / 200)', degree=1, Sc=Sc, Srange=Srange, tcd=tcd); # calculate profile
f['S_n'].assign(interpolate(S_init, f['S_n'].ufl_function_space()))
#feffi.plot.plot_single(f['S_n'], title='Salinity', display=True)

deltarho = interpolate(Expression('999.8*(1+gamma*(S)-beta*(T))-1000', rho_0=config['rho_0'], S_0=config['S_0'], T_0=config['T_0'], gamma=config['gamma'], beta=config['beta'], T=f['T_n'], S=f['S_n'], degree=1), f['S_n'].ufl_function_space())
feffi.plot.plot_single(deltarho, title='Density', display=True)
#return

domain = feffi.boundaries.Domain(
    fenics_mesh,
    f_spaces,
    boundaries = {
      'bottom' : feffi.boundaries.Bound_Bottom(fenics_mesh),
      'ice_shelf' : Bound_Ice_Shelf(),
      'left' : feffi.boundaries.Bound_Left(fenics_mesh),
      'right' : feffi.boundaries.Bound_Right(fenics_mesh),
      'top' : feffi.boundaries.Bound_Top(fenics_mesh),
    },
    BCs = feffi.parameters.config['BCs'])
#domain.show_boundaries() # with paraview installed, will show boundary markers

simulation = feffi.simulation.Simulation(f, domain)
simulation.run()
