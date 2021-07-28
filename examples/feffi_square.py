# feffi module lives one level up current dir
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import feffi
import os

# Importing feffi results in the default config file `square.yml` being
# parsed already. To tweak config, see feffi.parameters.define_parameters().
# All feffi functions take this config by default, but tweaks to the config
# can be made locally to each function by passing the relevant kwargs to
# each function call.
feffi.parameters.define_parameters({
    'config_file' : os.path.join('feffi', 'config', 'square.yml') })

# If run from console, accept commandline arguments
if __name__ == '__main__':
    feffi.parameters.parse_commandline_args()

# Create mesh over simulation domain
mesh = feffi.mesh.create_mesh()

# Define V, Q, T, S function spaces
f_spaces = feffi.functions.define_function_spaces(mesh)

# Define functions later used in variational forms
f = feffi.functions.define_functions(f_spaces)

# Initialize functions to closest steady state to speed up convergence
feffi.functions.init_functions(f)

# Define boundaries and boundary conditions as given in config
domain = feffi.boundaries.Domain(mesh, f_spaces)

# Initializes a feffi simulation
simulation = feffi.simulation.Simulation(f, domain)

# Run simulation until a stopping criteria is met. You may also advance by
# individual timesteps with `simulation.timestep()`
simulation.run()

# Plot mesh and solutions, displaying and saving them as png files
feffi.plot.plot_solutions(f, display=False)

feffi.flog.info('Plots can be found in {}'.format(feffi.parameters.config['plot_path']))
