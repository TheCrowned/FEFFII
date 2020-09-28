import feffi
import os, time, logging, fenics
from datetime import datetime
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

start_time = time.time()
logging.info('Started at %s ' % str(datetime.now()))

feffi.parameters.define_parameters(
	user_config={'final_time':20},
	config_file=os.path.join('config', 'default.yml')
)

if __name__ == '__main__':
	feffi.parameters.parse_commandline_args()

logging.info('Parameters are: ' + str(feffi.parameters.config))

mesh = feffi.mesh.create_mesh(domain='square')

f_spaces = feffi.functions.define_function_spaces(mesh)
f = feffi.functions.define_functions(f_spaces)
feffi.functions.init_functions(f)

feffi.functions.define_variational_problems(f, mesh)