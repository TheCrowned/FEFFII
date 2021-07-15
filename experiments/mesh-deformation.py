# feffi module lives one level up current dir
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import feffi
from fenics import *
import mshr
import matplotlib.pyplot as plt
import numpy as np
import imageio

feffi.parameters.define_parameters({
    'config_file' : 'feffi/config/lid-driven-cavity-3D.yml',
})
gif_filenames = []

# MESH INIT (unstructured)
points = [(0,0), (1,0), (1,1), (0, 1)]
Points = [Point(p) for p in points]
g2d = mshr.Polygon(Points)
g3d = mshr.Extrude2D(g2d, 1)
mesh = mshr.generate_mesh(g3d, 25)
mesh = UnitCubeMesh(10,10,10)

feffi.plot.plot_single(mesh, display=True, title="Starting mesh")
plot(mesh, title='Starting (ocean) mesh')
filename = 'mesh-1.png'
plt.savefig(filename, dpi=300)
plt.close()
for i in range(5):
    gif_filenames.append(filename)

# FEFFI INIT
f_spaces = feffi.functions.define_function_spaces(mesh)
f = feffi.functions.define_functions(f_spaces)
domain = feffi.boundaries.Domain(mesh, f_spaces)
simul = feffi.simulation.Simulation(f, domain.BCs)

# New "goal" left profile
# NOTE: goal profile is supposed to be COARSER than current profile
step_size = 0.1
goal_profile = np.array([(0.5*(y-0.5)**2-0.1, y, z)
                    for y in np.arange(min(mesh.coordinates()[:,1]),
                                       max(mesh.coordinates()[:,1])+step_size,
                                       step_size)
                    for z in np.arange(min(mesh.coordinates()[:,2]),
                                       max(mesh.coordinates()[:,2])+step_size,
                                       step_size)])
#print(goal_profile)
#fig = plt.figure(figsize = (10, 7))
#ax = plt.axes(projection ="3d")
#ax.scatter3D(goal_profile[:,0], goal_profile[:,1], goal_profile[:,2])
#plt.show()
total_steps = 5
for i in range(total_steps):
    simul.timestep()

    def_coeff = (i+1)/total_steps
    goal_profile_now = [(def_coeff*x, y, z)
                    for (x, y, z) in goal_profile]

    domain.deform_boundary('left', goal_profile_now)
    plot(mesh, title='Starting (ocean) mesh')
    plt.show()
    plt.close()

    plot(mesh, title='{} (ocean) vel'.format(i))
    filename = 'mesh{}.png'.format(i)
    plt.savefig(filename, dpi=300)
    plt.close()
    for i in range(3):
        gif_filenames.append(filename)

# build gif
with imageio.get_writer('mygif.gif', mode='I') as writer:
    for filename in gif_filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove files
#for filename in set(gif_filenames):
#    os.remove(filename)