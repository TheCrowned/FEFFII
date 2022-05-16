# feffii module lives one level up current dir
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import feffii
import os

# Importing feffii results in the default config file `square.yml` being
# parsed already. To tweak config, see feffii.parameters.define_parameters().
# All feffii functions take this config by default, but tweaks to the config
# can be made locally to each function by passing the relevant kwargs to
# each function call.
feffii.parameters.define_parameters({
    'config_file' : os.path.join('feffii', 'config', 'square.yml') })

# If run from console, accept commandline arguments
if __name__ == '__main__':
    feffii.parameters.parse_commandline_args()

# Create mesh over simulation domain
mesh = feffii.mesh.create_mesh()

# Define V, Q, T, S function spaces
f_spaces = feffii.functions.define_function_spaces(mesh)

# Define functions later used in variational forms
f = feffii.functions.define_functions(f_spaces)

# Initialize functions to closest steady state to speed up convergence
feffii.functions.init_functions(f)

# Define boundaries and boundary conditions as given in config
domain = feffii.boundaries.Domain(mesh, f_spaces)

# Initializes a feffii simulation
simulation = feffii.simulation.Simulation(f, domain)

# Run simulation until a stopping criteria is met. You may also advance by
# individual timesteps with `simulation.timestep()`
simulation.run()

# Plot mesh and solutions, displaying and saving them as png files
feffii.plot.plot_solutions(f, display=False)

feffii.flog.info('Plots can be found in {}'.format(feffii.parameters.config['plot_path']))
