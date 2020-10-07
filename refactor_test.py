import feffi
import os, logging

feffi.parameters.define_parameters(
    user_config = {
        'final_time' : 20,
        'config_file' : os.path.join('examples', 'config', 'square.yml')
    }
)

if __name__ == '__main__':
    feffi.parameters.parse_commandline_args()

logging.info('Parameters are: ' + str(feffi.parameters.config))

mesh = feffi.mesh.create_mesh()

f_spaces = feffi.functions.define_function_spaces(mesh)
f = feffi.functions.define_functions(f_spaces)
feffi.functions.init_functions(f)

(stiffness_mats, load_vectors) = feffi.functions.define_variational_problems(f, mesh)

domain = feffi.boundaries.Domain(mesh, f_spaces)

simulation = feffi.simulation.Simulation(f, stiffness_mats, load_vectors, domain.BCs)
simulation.run()

feffi.plot.plot_single(f['u_'], display = True)
#feffi.plot.plot_solutions(f, display = True)
