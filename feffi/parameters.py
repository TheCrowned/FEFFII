import os, yaml, logging, argparse, fenics, time

config = {}

def define_parameters(user_config={}):
    """Defines running parameters for model.

    Parameters are stored in a global variable named `config`.
    The default location for config files is in a `config` dir located
    just out of the model code dir (i.e. `feffi/../config/`).
    A config file is first used to load configuration and, if none is
    given/found, a default one is used.
    If a non-empty dictionary is supplied as well, its entries will
    complete/overwrite the ones coming from the config.

    Notice that just the act of `import feffi` already sets up a default
    configuration.

    Parameters
    ----------
    user_config : dictionary, optional
            If non-empty, entries from this dictionary have priority in
            parameters definition.
            Can contain a `config_file` entry pointing to a YAML file
            containing configuration (either absolute or relative path).

    Examples
    ----------
    1)  Use config file located in ../config/default.yml,
        but using `20` as value for `final_time`.

        feffi.parameters.define_parameters(
            user_config = {
                'final_time' : 20,
                'config_file' : os.path.join('config', 'default.yml')
            }
        )
    """

    global config

    # Double call to dirname() reaches parent dir of current script
    parent_dir = os.path.dirname(os.path.dirname(__file__))

    config_file_path = os.path.join(parent_dir, 'examples', 'config', 'square.yml')

    # If given, open custom config file...
    if(user_config.get('config_file') != None and user_config['config_file'] != ''):
        config_file_path = os.path.join(parent_dir, user_config['config_file']) if not os.path.isabs(user_config['config_file']) else user_config['config_file']

        if(not os.path.isfile(config_file_path)):
            logging.warning('Given config file does not exist -- fallbacking to default one.')
            config_file_path = os.path.join(parent_dir, 'config', 'default.yml')

            if(not os.path.isfile(config_file_path)):
                logging.error('No valid config file found. Make sure to pass ALL required config arguments as dict.')

    config = yaml.safe_load(open(config_file_path))
    config['config_file'] = config_file_path

    # If (some) dictionary config is provided as class init parameter, that overwrites the default config
    # Purge None entries before we merge. This is needed because if you provide a --config-file from commandline,
    # then the defaults None would overwrite previous settings.
    if isinstance(user_config, dict):
        purged_user_config = {key : val for key, val in user_config.items() if val != None}
        config.update(purged_user_config)
    else:
        logging.warning('Supplied non-dictionary user config')

    # Set some further entries
    label = " --label " + config['label'] if config['label'] else ""
    config['plot_path'] = os.path.join(parent_dir, 'plots', '%d --final-time %.0f --steps-n %d --mesh-resolution %d%s/' % (round(time.time()), config['final_time'], config['steps_n'], config['mesh_resolution'], label))

def parse_commandline_args():
    """Provides support for command line arguments through argparse.

    Values passed from commandline have priority over both config files
    and custom user dictionary config.
    """

    '''def add_bool_arg(parser, name, default=False, dest=None, help=None):
        """Shortcut for nice boolean commandline arguments.

        From https://stackoverflow.com/questions/53937481/python-command-line-arguments-foo-and-no-foo
        """

        if(dest is None):
            dest = name

        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=dest, action='store_true', help=help)
        group.add_argument('--no-' + name, dest=dest, action='store_false', help=help)
        parser.set_defaults(**{name:default})'''

    global config

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', default=config['config_file'], dest='config_file', help='File from which to read config parameters')
    parser.add_argument('--final-time', type=float, dest='final_time', help='How long to run the simulation for (hours) (default: %(default)s)')
    parser.add_argument('--steps-n', type=int, dest='steps_n', help='How many steps each of the "seconds" is made of (default: %(default)s)')
    parser.add_argument('--precision', type=int, dest='simulation_precision', help='Precision at which converge is achieved, for all variables (power of ten) (default: %(default)s)')
    parser.add_argument('--nu', type=float, dest='nu', nargs="*", help='Viscosity, m^2/s. Expects 1, 2 or 4 space-separated entries, depending on whether a scalar, vector or tensor is wished (default: %(default)s)')
    parser.add_argument('--alpha', type=float, dest='alpha', nargs="*", help='Diffusivity coefficient for temperature/salinity, m^2/s. Expects 1, 2 or 4 space-separated entries, depending on whether a scalar, vector or tensor is wished (default: %(default)s)')
    parser.add_argument('--rho-0', type=float, dest='rho_0', help='Density, kg/m^3 (default: %(default)s)')
    parser.add_argument('--beta', type=float, help='Water thermal expansion coefficient, 1/°C (default: %(default)s)')
    parser.add_argument('--gamma', type=float, help='Water salinity expansion coefficient, 1/PSU (default: %(default)s)')
    parser.add_argument('--T-0', type=float, dest='T_0', help='Reference temperature, °C (default: %(default)s)')
    parser.add_argument('--S-0', type=float, dest='S_0', help='Reference salinity, PSU (default: %(default)s)')
    parser.add_argument('--domain', help='What domain to use, either `square` (1km x 1km) or `custom` (default: %(default)s)')
    parser.add_argument('--domain-size-x', type=int, dest='domain_size_x', help='Size of domain in x direction (i.e. width) (default: %(default)s)')
    parser.add_argument('--domain-size-y', type=int, dest='domain_size_y', help='Size of domain in y direction (i.e. height) (default: %(default)s)')
    parser.add_argument('--shelf-size-x', type=float, dest='shelf_size_x', help='Size of ice shelf in x direction (i.e. width) (default: %(default)s)')
    parser.add_argument('--shelf-size-y', type=float, dest='shelf_size_y', help='Size of ice shelf in y direction (i.e. height) (default: %(default)s)')
    parser.add_argument('--mesh-resolution', type=int, dest='mesh_resolution', help='Mesh resolution (default: %(default)s) - does not apply to `rectangle` domain')
    parser.add_argument('--mesh-resolution-x', type=int, dest='mesh_resolution_x', help='Mesh resolution in x direction (default: %(default)s) - only applies to `rectangle` domain')
    parser.add_argument('--mesh-resolution-y', type=int, dest='mesh_resolution_y', help='Mesh resolution in y direction (default: %(default)s) - only applies to `rectangle` domain')
    parser.add_argument('--mesh-resolution-sea-top-y', type=int, dest='mesh_resolution_sea_y', help='Mesh resolution for sea top beside ice shelf in y direction (default: %(default)s) - only applies to `rectangle` domain')
    parser.add_argument('--store-sol', dest='store_solutions', action='store_true', help='Whether to save iteration solutions for display in Paraview (default: %(default)s)')
    parser.add_argument('--label', help='Label to append to plots folder (default: %(default)s)')
    parser.add_argument('--max-iter', type=int, dest='max_iter', help='Stop simulation after given number of timesteps; 0 = infinite (default: %(default)s)')
    #parser.add_argument('-v', '--verbose', default=config['verbose'], dest='verbose', action='store_true', help='Whether to display debug info (default: %(default)s)')
    #parser.add_argument('-vv', '--very-verbose', default=config['very_verbose'], dest='very_verbose', action='store_true', help='Whether to display mroe debug info (default: %(default)s)')
    #add_bool_arg(parser, 'plot', default=config['plot'], help='Whether to plot solution (default: %(default)s)')

    commandline_args = parser.parse_args()
    commandline_args_dict = {arg: getattr(commandline_args, arg) for arg in vars(commandline_args)}

    # Empty current config and set anew. Should make troubleshooting easier, since we don't want the config to be already populated if a custom config is provided
    config = {}
    define_parameters(commandline_args_dict)

def assemble_viscosity_tensor(visc):
    """Creates a proper viscosity tensor given relevant values.
    Notice that input must always be a list, even if it has only one element.

    Parameters
    ----------
    visc : list (of floats)
            If 1 entry is given, value will be used for all diagonal entries (other entries = 0).
            If 2 entries are given, they will be used for diagonal entries (other entries = 0).
            If 4 entries are given, they will compose the full tensor.

    Examples
    ----------
    1) Obtain a tensor of the form
       (a  0
        0  a)

       assemble_viscosity_tensor([a])

    2) Obtain a tensor of the form
       (a  0
        0  b)

       assemble_viscosity_tensor([a, b])

    3) Obtain a tensor of the form
       (a  b
        c  d)

       assemble_viscosity_tensor([a, b, c, d])
    """

    visc = visc#[i*0.0036 for i in visc] # from m^2/s to km^2/h

    if len(visc) == 1:
        output = fenics.as_tensor((
            (visc[0], 0),
            (0, visc[0])
        ))

    elif len(visc) == 2:
        output = fenics.as_tensor((
            (visc[0], 0),
            (0, visc[1])
        ))

    elif len(visc) == 4:
        output = fenics.as_tensor((
            (visc[0], visc[1]),
            (visc[2], visc[3])
        ))

    else:
        raise ValueError("Viscosity needs 1, 2 or 4 entries input, %d given" % len(visc))

    return output