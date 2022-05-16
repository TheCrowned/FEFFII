# feffi module lives one level up current dir
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, '/home/stefano/git/fenics-phd/')

# Force monothread in OpenBLAS, seems to even speed up
# https://stackoverflow.com/questions/52026652/openblas-blas-thread-init-pthread-create-resource-temporarily-unavailable
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import feffii
from fenics import *
import numpy as np
import pygmsh


####################
### Config setup ###
####################

feffii.parameters.define_parameters({
    'config_file' : 'feffii/config/ryder-setup.yml',
})
config = feffii.parameters.config

##############################
### Set up geometry + mesh ###
##############################

x, bed, ice_thickness, ice_surface, ice_profile = [], [], [], [], []

finput = open('/home/stefano/git/fenics-phd/experiments/sherard-osborn/RyderLonger_dx200m_0001_Input.txt', 'r')
data = finput.readlines()
for line in data:
    values = line.split(' ')

    # before x = 280k it is ice only; grounding line is at ~285k
    # if ice profile is within 5 meters from bed, it means we still haven't reached grounding line
    if float(values[3])-float(values[2]) - float(values[1]) < 5:
        continue
        
    x.append(float(values[0]))
    bed.append(float(values[1]))
    ice_thickness.append(float(values[2]))
    ice_surface.append(float(values[3]))
    ice_profile.append(float(values[3])-float(values[2]))

# rescale so that origin is at (0,0)
min_x = min(x)
min_bed = min(bed)
min_ice_profile = min(ice_profile)
x = [el-min_x for el in x]
bed = [el-min_bed for el in bed]
ice_profile = [el-min_bed for el in ice_profile]
# find entry when ice profile becomes flat and cut list there
for idx in range(len(ice_profile)):
    if x[idx] > 28000 and near(ice_profile[idx], ice_profile[idx+1]):
        ice_profile = ice_profile[0:idx]
        break
top_right = (max(x), max(ice_profile), 0)

# build points list
points =  list(zip(x, bed, [0]*len(x)))                          # bottom
points += [top_right]                                            # left
points += list(reversed(list(zip(x, ice_profile, [0]*len(x)))))  # ice shelf

##############################
### PyGMSH mesh generation ###
##############################

fenics_mesh = False
with pygmsh.geo.Geometry() as geom:
    poly = geom.add_polygon(points, mesh_size=config['mesh_resolution'])

    # Generate mesh
    mesh = geom.generate_mesh()
    mesh.write('mesh.xdmf')
    fenics_mesh = feffii.mesh.pygmsh2fenics_mesh(mesh)
    #feffii.plot.plot_single(fenics_mesh, display=True)
    print(fenics_mesh.num_vertices())
    print(fenics_mesh.hmin())
    print(fenics_mesh.hmax())

# Ice shelf boundary
class Bound_Ice_Shelf(SubDomain):        
    def inside(self, p, on_boundary):
        #return True
        #if p[0] > 29000:
        #    return False
        #else:
        #    return True

        closest_neighbor = feffi.boundaries.closest_k_nodes(p, list(zip(x, ice_profile)), 1)
        print(list(closest_neighbor.items()))
        closest_neighbor = list(closest_neighbor.items())[0][1]
        closest_neighbor_coord = closest_neighbor['coord']
        closest_neighbor_dist = closest_neighbor['dist']
        print(p, closest_neighbor)
        v =  (closest_neighbor_dist == 0 and on_boundary)
        print(v)
        return v


######################
## Simulation setup ##
######################

f_spaces = feffii.functions.define_function_spaces(fenics_mesh)
f = feffii.functions.define_functions(f_spaces)
#feffii.functions.init_functions(f)

# Initial conditions for T,S that mimick Ryder glacier
tcd = -0.8; # Thermocline depth
Tmin, Tmax = -1.6, 0.4; # minimum/maximum Temperature
Tc = (Tmax + Tmin) / 2; # middle temperature
Trange = Tmax - Tmin; # temperature range
T_init = Expression('Tc - Trange / 2 * -tanh(pi * 1000*(-x[1] - tcd) / 200)', degree=1, Tc=Tc, Trange=Trange, tcd=tcd); # calculate profile
f['T_n'].assign(interpolate(T_init, f['T_n'].ufl_function_space()))
f['T_'].assign(interpolate(T_init, f['T_n'].ufl_function_space()))
#feffii.plot.plot_single(f['T_n'], title='Temperature', display=True)

Sc = 34.5;
Srange = -1;
S_init = Expression('Sc + Srange / 2 * -tanh(pi * 1000*(-x[1] - tcd) / 200)', degree=1, Sc=Sc, Srange=Srange, tcd=tcd); # calculate profile
f['S_n'].assign(interpolate(S_init, f['S_n'].ufl_function_space()))
f['S_'].assign(interpolate(S_init, f['S_n'].ufl_function_space()))
#feffii.plot.plot_single(f['S_n'], title='Salinity', display=True)

deltarho = interpolate(Expression('999.8*(1+gamma*(S)-beta*(T))-1000', rho_0=config['rho_0'], S_0=config['S_0'], T_0=config['T_0'], gamma=config['gamma'], beta=config['beta'], T=f['T_n'], S=f['S_n'], degree=1), f['S_n'].ufl_function_space())
#feffii.plot.plot_single(deltarho, title='Density', display=True)
#return

domain = feffii.boundaries.Domain(
    fenics_mesh,
    f_spaces,
    boundaries = {
      'bottom' : feffii.boundaries.Bound_Bottom(fenics_mesh),
      'ice_shelf' : Bound_Ice_Shelf(x, ice_profile),
      'left' : feffii.boundaries.Bound_Left(fenics_mesh),
      'right' : feffii.boundaries.Bound_Right(fenics_mesh),
      'top' : feffii.boundaries.Bound_Top(fenics_mesh),
    },
    BCs = feffii.parameters.config['BCs'])
domain.show_boundaries() # with paraview installed, will show boundary markers

#simulation = feffii.simulation.Simulation(f, domain)
#simulation.run()
