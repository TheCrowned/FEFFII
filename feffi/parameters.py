import os, yaml, logging, argparse

config = {}

def define_parameters(user_config={}, config_file=''):
	"""Defines running parameters for model.

	Parameters are stored in a global variable named `config`.
	The default location for config files is in a `config` dir located just out of the model code dir (i.e. `feffi/../config/`).
	A config file is first used to load configuration and, if none is given/found, a default one is used.
	If a non-empty dictionary is supplied as well, its entries will complete/overwrite the ones coming from the config.

        Parameters
        ----------
        user_config : dictionary, optional
                If non-empty, entries from this dictionary have priority in parameters definition.
        config_file : string, optional
                YAML file containing configuration. Accepts either an absolute path or, if a relative path is provided, it assumes config files are located in ../config.

        Examples
        ----------
        1) Use config file located in ../config/default.yml, but using `20` as value for `final_time`.

        feffi.parameters.define_parameters(
            user_config = {'final_time' : 20},
            config_file = os.path.join('config', 'default.yml')
        )
    """

	global config

	# Double call to dirname() reaches parent dir of current script
	dirname = os.path.dirname(os.path.dirname(__file__))

	# If given, open custom config file...
	if(config_file != ''):
		config_file_path = os.path.join(dirname, config_file) if not os.path.isabs(config_file) else config_file

		if(os.path.isfile(config_file_path)):
			config = yaml.safe_load(open(config_file_path))
		else:
			logging.warning('Given config file does not exist -- fallbacking to default one.')

	# ...fallback to default one otherwise
	config_file_path = os.path.join(dirname, 'config', 'default.yml')
	if(config == {} and os.path.isfile(config_file_path)):
		config = yaml.safe_load(open(config_file_path))
	elif(config == {}):
		logging.error('No valid config file found. Make sure to pass ALL required config arguments as dict.')

	# If (some) dictionary config is provided as class init parameter, that overwrites the default config
	if isinstance(user_config, dict):
		config.update(user_config)
	else:
		logging.warning('Supplied non-dictionary user config')

def parse_commandline_args():
	"""Provides support for command line arguments through argparse.

	Values passed from commandline have priority over both config files and custom user dictionary config.
    """

	def add_bool_arg(parser, name, default=False, dest=None, help=None):
		"""Shortcut for nice boolean commandline arguments.

		From https://stackoverflow.com/questions/53937481/python-command-line-arguments-foo-and-no-foo
		"""

		if(dest is None):
			dest = name

		group = parser.add_mutually_exclusive_group(required=False)
		group.add_argument('--' + name, dest=dest, action='store_true', help=help)
		group.add_argument('--no-' + name, dest=dest, action='store_false', help=help)
		parser.set_defaults(**{name:default})

	global config

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
	parser.add_argument('-vv', '--very-verbose', default=config['very_verbose'], dest='very_verbose', action='store_true', help='Whether to display mroe debug info (default: %(default)s)')
	add_bool_arg(parser, 'plot', default=config['plot'], help='Whether to plot solution (default: %(default)s)')

	commandline_args = parser.parse_args()
	commandline_args_dict = {arg: getattr(commandline_args, arg) for arg in vars(commandline_args)}

	# Empty current config and set anew. Should make troubleshooting easier, since we don't want the config to be already populated if a custom config is provided
	config = {}
	define_parameters(commandline_args_dict, commandline_args_dict['config_file'])
