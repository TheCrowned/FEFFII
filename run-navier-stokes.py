from navierstokes import NavierStokes
from fenics import *
import math
import time, os
import pathlib
from datetime import datetime
from importlib import import_module

ns = NavierStokes({})

#args = ns.parse_commandline_args()

#print(ns.args)
#args_dict = {arg: getattr(args, arg) for arg in vars(args)}

sill = {}
#final_time = float(args.final_time)     				# final time
#num_steps = int(args.steps_n)   						# number of time steps per time unit
#dt_scalar = 1 / num_steps 								# time step size
#simulation_precision = int(args.simulation_precision)	# precision at which converge is considered to be achieved (for all variables)
#mu_scalar = float(args.viscosity)      				# kinematic viscosity

# Values used in variational forms, some provided as input from terminal
#nu = Constant(args.nu)
#rho_0 = Constant(args.rho_0) #https://en.wikipedia.org/wiki/Seawater#/media/File:WaterDensitySalinity.png
#g = 1.27*10**5
#alpha = Constant(10**(-4))
#beta = Constant(7.6*10**(-4))
#T_0 = Constant(1)
#S_0 = Constant(35)

start_time = time.time()
ns.log('--- Started at %s --- ' % str(datetime.now()), True)

ns.log('--- Parameters are: ---', True)
ns.log(str(ns.args), True)

if(ns.args.very_verbose == True):
	set_log_active(True)
	set_log_level(1)

ns.create_mesh()
ns.define_function_spaces()

start_time = time.time()

ns.define_variational_problems()
ns.boundary_conditions()
ns.mesh_add_sill(ns.args.domain_size_x/2, ns.args.domain_size_y/5, ns.args.domain_size_x/5)
ns.run_simulation()

ns.log('--- Finished at %s --- ' % str(datetime.now()), True)
ns.log('--- Duration: %s seconds --- ' % round((time.time() - start_time), 2), True)

if(ns.args.plot == True):
	ns.plot_solution()

os.system('xdg-open "' + ns.plot_path + '"')

