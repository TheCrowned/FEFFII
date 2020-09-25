import os, yaml, logging, argparse

config = {}

def define_parameters(user_config={}, config_file=''):
	global config
	
	#Open custom config file if given, or fallback to default one
	if(config_file != ''):
		if(os.path.isfile(config_file)):
			config = yaml.safe_load(open(config_file))
		else:
			logging.warning('Given config file does not exist -- fallbacking to default one.')
	
	if(config == {} and os.path.isfile('config/default.yml')):
		config = yaml.safe_load(open('config/default.yml'))
	elif(config == {}):
		logging.error('No valid config file found. Make sure to pass ALL required config arguments as dict.')

	#If (some) config is provided as class init parameter, that overwrites the default config
	config.update(user_config)

def parse_commandline_args():
	global config
	
	# From https://stackoverflow.com/questions/53937481/python-command-line-arguments-foo-and-no-foo
	def add_bool_arg(parser, name, default=False, dest=None, help=None):
		if(dest is None):
			dest = name

		group = parser.add_mutually_exclusive_group(required=False)
		group.add_argument('--' + name, dest=dest, action='store_true', help=help)
		group.add_argument('--no-' + name, dest=dest, action='store_false', help=help)
		parser.set_defaults(**{name:default})

	# See https://docs.python.org/2/library/argparse.html
	parser = argparse.ArgumentParser()
	parser.add_argument('--config-file', default='config/default.yml', dest='config_file', help='File from which to read config parameters')
	parser.add_argument('--final-time', default=config['final_time'], type=float, dest='final_time', help='How long to run the simulation for (hours) (default: %(default)s)')
	parser.add_argument('--steps-n', default=config['steps_n'], type=int, dest='steps_n', help='How many steps each of the "seconds" is made of (default: %(default)s)')
	parser.add_argument('--precision', default=config['precision'], type=int, dest='simulation_precision', help='Precision at which converge is achieved, for all variables (power of ten) (default: %(default)s)')
	parser.add_argument('--viscosity', default=config['nu'], type=float, dest='nu', nargs="*", help='Viscosity, m^2/s. Expects 1, 2 or 4 space-separated entries, depending on whether a scalar, vector or tensor is wished (default: %(default)s)')
	parser.add_argument('--rho-0', default=config['rho_0'], type=float, dest='rho_0', help='Density, kg/m^3 (default: %(default)s)')
	parser.add_argument('--alpha', default=config['alpha'], type=float, help='Water thermal expansion coefficient, 1/°C (default: %(default)s)')
	parser.add_argument('--beta', default=config['beta'], type=float, help='Water salinity expansion coefficient, 1/PSU (default: %(default)s)')
	parser.add_argument('--T-0', default=config['T_0'], type=float, dest='T_0', help='Reference temperature, °C (default: %(default)s)')
	parser.add_argument('--S-0', default=config['S_0'], type=float, dest='S_0', help='Reference salinity, PSU (default: %(default)s)')
	parser.add_argument('--ocean-bc', default=config['ocean_bc'], dest='ocean_bc', help='Regulates in/out flow at ocean boundary. If a number is given, it will be used as scaling-coefficient of the sinusodial BC on ocean boundary. If a string is given, it will be used as formula for the ocean BC (default: %(default)s)')
	parser.add_argument('--domain', default=config['domain'], help='What domain to use, either `square` (1km x 1km) or `custom` (default: %(default)s)')
	parser.add_argument('--domain-size-x', default=config['domain_size_x'], type=int, dest='domain_size_x', help='Size of domain in x direction (i.e. width) (default: %(default)s)')
	parser.add_argument('--domain-size-y', default=config['domain_size_y'], type=int, dest='domain_size_y', help='Size of domain in y direction (i.e. height) (default: %(default)s)')
	parser.add_argument('--shelf-size-x', default=config['shelf_size_x'], type=float, dest='shelf_size_x', help='Size of ice shelf in x direction (i.e. width) (default: %(default)s)')
	parser.add_argument('--shelf-size-y', default=config['shelf_size_y'], type=float, dest='shelf_size_y', help='Size of ice shelf in y direction (i.e. height) (default: %(default)s)')
	parser.add_argument('--mesh-resolution', default=config['mesh_resolution'], type=int, dest='mesh_resolution', help='Mesh resolution (default: %(default)s) - does not apply to `rectangle` domain')
	parser.add_argument('--mesh-resolution-x', default=config['mesh_resolution_x'], type=int, dest='mesh_resolution_x', help='Mesh resolution in x direction (default: %(default)s) - only applies to `rectangle` domain')
	parser.add_argument('--mesh-resolution-y', default=config['mesh_resolution_y'], type=int, dest='mesh_resolution_y', help='Mesh resolution in y direction (default: %(default)s) - only applies to `rectangle` domain')
	parser.add_argument('--mesh-resolution-sea-top-y', default=config['mesh_resolution_sea_top_y'], type=int, dest='mesh_resolution_sea_y', help='Mesh resolution for sea top beside ice shelf in y direction (default: %(default)s) - only applies to `rectangle` domain')
	parser.add_argument('--store-sol', default=config['store_sol'], dest='store_solutions', action='store_true', help='Whether to save iteration solutions for display in Paraview (default: %(default)s)')
	parser.add_argument('--label', default='', help='Label to append to plots folder (default: %(default)s)')
	parser.add_argument('-v', '--verbose', default=config['verbose'], dest='verbose', action='store_true', help='Whether to display debug info (default: %(default)s)')
	parser.add_argument('-vv', '--very-verbose', default=config['very_verbose'], dest='very_verbose', action='store_true', help='Whether to display debug info from FEniCS as well (default: %(default)s)')
	add_bool_arg(parser, 'plot', default=config['plot'], help='Whether to plot solution (default: %(default)s)')

	commandline_args = parser.parse_args()
	commandline_args_dict = {arg: getattr(commandline_args, arg) for arg in vars(commandline_args)}

	define_parameters(commandline_args_dict, commandline_args_dict['config_file'])
