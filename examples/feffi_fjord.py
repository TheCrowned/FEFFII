# feffi module lives one level up current dir
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import feffi
import os
import fenics

# Define parameters using custom config file and providing some parameters
# as dictionary entries
feffi.parameters.define_parameters(
    user_config = {
        'config_file' : os.path.join('feffi', 'config', 'fjord.yml')})

# If run from console, accept commandline arguments
'''if __name__ == '__main__':
    feffi.parameters.parse_commandline_args()'''

# Create mesh over simulation domain
mesh = feffi.mesh.create_mesh()

# Define V, Q, T, S function spaces
f_spaces = feffi.functions.define_function_spaces(mesh)

# Define functions later used in variational forms
f = feffi.functions.define_functions(f_spaces)

# Initialize functions to closest steady state to speed up convergence
feffi.functions.init_functions(f)

# Define boundaries and boundary conditions as given in config
domain = feffi.boundaries.Domain(mesh, f_spaces, f)

# Initializes a feffi simulation
simulation = feffi.simulation.Simulation(f, domain.BCs)

# Run simulation until a stopping criteria is met. You may also advance by
# individual timesteps with `simulation.timestep()`
simulation.run()

# Export solutions for comparison
fenics.File('test_u_{}.xml'.format(feffi.parameters.config['beta'])) << f['u_']
fenics.File('test_u_{}.xml'.format(feffi.parameters.config['beta'])) << f['p_']
fenics.File('test_u_{}.xml'.format(feffi.parameters.config['beta'])) << f['T_']

# Plot mesh and solutions, saving them as png files
feffi.plot.plot_single(mesh, file_name = 'mesh.png', title = 'Mesh', display = False)
feffi.parameters.convert_from_ms_to_kmh_output(f)
feffi.plot.plot_solutions(f, display = False)

feffi.flog.info('Plots can be found in %s' % feffi.parameters.config['plot_path'])
