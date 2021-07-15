import feffi
from fenics import *
import mshr
import matplotlib.pyplot as plt
import numpy as np

feffi.parameters.define_parameters({
    'config_file' : 'feffi/config/lid-driven-cavity-3D.yml',
})

#points = [(0,0), (4,0), (4.5,0.2), (5, 0.5), (5.5, 0.2), (6,0), (10,0), (10,1), (2,1), (2,0.8), (0, 0.8)]
points = [(0,0), (1,0), (1,1), (0, 1)]
Points = [Point(p) for p in points]
g2d = mshr.Polygon(Points)
g3d = mshr.Extrude2D(g2d, 1)
m = mshr.generate_mesh(g3d, 10)
#m.coordinates()[:] = np.array([m.coordinates()[:, 0], m.coordinates()[:, 2], m.coordinates()[:, 1]]).transpose()
#plot(m)
#plt.show()

f_spaces = feffi.functions.define_function_spaces(m)
f = feffi.functions.define_functions(f_spaces)
domain = feffi.boundaries.Domain(m, f_spaces)