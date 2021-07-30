# feffi module lives one level up current dir
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from feffi import *
import fenics
import matplotlib.pyplot as plt

parameters.define_parameters({
    'config_file' : 'feffi/config/buoyancy-driven-cavity.yml',
    'max_iter' : 3000
})
parameters.parse_commandline_args()

mesh = mesh.create_mesh()
f_spaces = functions.define_function_spaces(mesh)
f = functions.define_functions(f_spaces)
functions.init_functions(f) # Init functions to closest steady state
domain = boundaries.Domain(mesh, f_spaces)

simul = simulation.Simulation(f, domain)
simul.run()

plot.plot_solutions(f, display=False)