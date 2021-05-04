import os
import yaml
import logging
import argparse
import fenics
import time
from pathlib import Path
from .functions import define_function_spaces, define_functions
from .boundaries import Domain

flog = logging.getLogger('feffi')
config = {}


def define_parameters(user_config={}):
    """Defines running parameters for model.

    Parameters are stored in a global variable named `config`.
    The default location for config files is in a `config` dir located
    inside the model code dir (i.e. `feffi/config/`).
    Takes a default config file to begin with and overwrites any of its
    entries which are also supplied through a user-given config file or dict.

    Remarks:
    1) Just the act of `import feffi` already sets up a default
    configuration.
    2) Errors/warnings happening in this function are NOT regularly
    logged. They are only printed. This is because init_logging is called after
    parameters have been defined (since plot_path is not set earlier).
    3) Config file feffi/config/default.yml should contain ALL config entries.

    Parameters
    ----------
    user_config : dictionary, optional
            If non-empty, entries from this dictionary have priority in
            parameters definition.
            Can contain a `config_file` entry pointing to a YAML file
            containing configuration (either absolute or relative path).

    Examples
    ----------
    1)  Use config file located in feffi/config/square.yml,
        but using `20` as value for `final_time`.

        feffi.parameters.define_parameters(
            user_config = {
                'final_time' : 20,
                'config_file' : os.path.join('feffi', 'config', 'square.yml')
            }
        )
    """

    global config

    # Double call to dirname() reaches parent dir of current script
    parent_dir = os.path.dirname(os.path.dirname(__file__))

    config_file_path = os.path.join('feffi', 'config', 'default.yml')

    # Load default config file
    # These values will be overwritten by later-fetched user config
    if(not os.path.isfile(config_file_path)):
        print(
            'Default config file not found. Things are likely to break.'
            'Make sure to pass ALL required config arguments as dict '
            'or through another config file.')
    else:
        config = yaml.safe_load(open(config_file_path))
        config['config_file'] = config_file_path

    # If given, open custom config file...
    if user_config.get('config_file'):
        if os.path.isabs(user_config['config_file']):
            config_file_path = user_config['config_file']
        else:
            config_file_path = os.path.join(
                parent_dir,
                user_config['config_file'])

        if(not os.path.isfile(config_file_path)):
            print(
                'Config file: \n{}\n does not exist\n'
                'Falling back to default one.'.format(config_file_path))
        else:
            config.update(yaml.safe_load(open(config_file_path)))
            config['config_file'] = config_file_path

    # If (some) dictionary config is provided as class init parameter,
    # that overwrites the default config.
    if isinstance(user_config, dict):
        config.update(user_config)
    else:
        print('Supplied non-dictionary user config, ignoring')

    # Define and create plot path in case not already given by config file
    if(config.get('plot_path') == None or len(config['plot_path']) == 0):
        config['plot_path'] = os.path.join(
            parent_dir,
            'plots',
            '{}'.format(round(time.time())))

    # If a relative path was given as config, make it relative to plots folder
    if not os.path.isabs(config['plot_path']):
        config['plot_path'] = os.path.join(
            parent_dir,
            'plots',
            config['plot_path'])

    Path(config['plot_path']).mkdir(parents=True, exist_ok=True)

    if config.get('convert_from_ms_to_kmh'):
        convert_from_ms_to_kmh_input(config)

    init_logging()


def init_logging():
    """
    Initialize feffi logger. This happens after parameters have been defined.

    Includes a stream handler to display logging on the terminal and a file
    handler to log messages to file. File logging goes into a simulation.log
    file located in config['plot_path'].
    """

    feffi_log_level = logging.DEBUG if config['very_verbose'] else logging.INFO

    flog.setLevel(feffi_log_level)
    # dark magic https://stackoverflow.com/a/44426266 to avoid messages showing up multiple times
    flog.propagate = False

    # If some handlers are already present, delete them. This is because
    # init_logging is called by define_parameters(), which is called multiple times.
    if len(flog.handlers) != 0:
        flog.handlers = []

    log_format = logging.Formatter(
        '%(asctime)s %(message)s', datefmt='[%H:%M:%S]')

    # Create two handlers:
    # one for file logging, another for terminal (stream) logging
    fh = logging.FileHandler(
        os.path.join(config['plot_path'], 'simulation.log'),
        mode='w',
        encoding='utf-8')
    fh.setLevel(feffi_log_level)
    fh.setFormatter(log_format)
    logging.getLogger('feffi').addHandler(fh)

    th = logging.StreamHandler()
    th.setLevel(feffi_log_level)
    th.setFormatter(log_format)
    logging.getLogger('feffi').addHandler(th)

    # Reduce FEniCS logging
    if config['very_verbose']:
        fenics.set_log_level(logging.WARNING)  # default is INFO
    else:
        fenics.set_log_level(logging.ERROR)
    #logging.getLogger('UFL').setLevel(logging.WARNING)
    #logging.getLogger('FFC').setLevel(logging.WARNING)


def parse_commandline_args():
    """Provides support for command line arguments through argparse.

    Values passed from commandline have priority over both config files
    and custom user dictionary config.
    """

    '''def add_bool_arg(parser, name, default=False, dest=None, help=None):
        """Shortcut for nice boolean commandline arguments.

        From
        https://stackoverflow.com/questions/53937481/
    python-command-line-arguments-foo-and-no-foo
        """

        if(dest is None):
            dest = name

        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name,
                dest=dest, action='store_true', help=help)
        group.add_argument('--no-' + name,
            dest=dest, action='store_false', help=help)
        parser.set_defaults(**{name:default})'''

    global config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file',
        default=config['config_file'],
        dest='config_file',
        help='File from which to read config parameters')
    parser.add_argument(
        '--final-time',
        type=float,
        dest='final_time',
        help='How long to run the simulation for (hours) (default: %(default)s)')
    parser.add_argument(
        '--steps-n',
        type=int,
        dest='steps_n',
        help='How many steps each of the "seconds" is made of (default: %(default)s)')
    parser.add_argument(
        '--precision',
        type=int,
        dest='simulation_precision',
        help='Precision at which converge is achieved, for all variables (power of ten) (default: %(default)s)')
    parser.add_argument(
        '--nu',
        type=float,
        dest='nu',
        nargs="*",
        help='Viscosity, m^2/s. Expects 1, 2 or 4 space-separated entries, depending on whether a scalar, vector or tensor is wished (default: %(default)s)')
    parser.add_argument(
        '--alpha',
        type=float,
        dest='alpha',
        nargs="*",
        help='Diffusivity coefficient for temperature/salinity, m^2/s. Expects 1, 2 or 4 space-separated entries, depending on whether a scalar, vector or tensor is wished (default: %(default)s)')
    parser.add_argument(
        '--rho-0',
        type=float,
        dest='rho_0',
        help='Density, kg/m^3 (default: %(default)s)')
    parser.add_argument(
        '--beta',
        type=float,
        help='Water thermal expansion coefficient, 1/°C (default: %(default)s)')
    parser.add_argument(
        '--gamma',
        type=float,
        help='Water salinity expansion coefficient, 1/PSU (default: %(default)s)')
    parser.add_argument(
        '--T-0',
        type=float,
        dest='T_0',
        help='Reference temperature, °C (default: %(default)s)')
    parser.add_argument(
        '--S-0',
        type=float,
        dest='S_0',
        help='Reference salinity, PSU (default: %(default)s)')
    parser.add_argument(
        '--domain',
        help='What domain to use, either `square` (1km x 1km) or `custom` (default: %(default)s)')
    parser.add_argument(
        '--domain-size-x',
        type=int,
        dest='domain_size_x',
        help='Size of domain in x direction (i.e. width) (default: %(default)s)')
    parser.add_argument(
        '--domain-size-y',
        type=int,
        dest='domain_size_y',
        help='Size of domain in y direction (i.e. height) (default: %(default)s)')
    parser.add_argument(
        '--shelf-size-x',
        type=float,
        dest='shelf_size_x',
        help='Size of ice shelf in x direction (i.e. width) (default: %(default)s)')
    parser.add_argument(
        '--shelf-size-y',
        type=float,
        dest='shelf_size_y',
        help='Size of ice shelf in y direction (i.e. height) (default: %(default)s)')
    parser.add_argument(
        '--mesh-resolution',
        type=int,
        dest='mesh_resolution',
        help='Mesh resolution (default: %(default)s) - does not apply to `rectangle` domain')
    parser.add_argument(
        '--mesh-resolution-x',
        type=int,
        dest='mesh_resolution_x',
        help='Mesh resolution in x direction (default: %(default)s) - only applies to `rectangle` domain')
    parser.add_argument(
        '--mesh-resolution-y',
        type=int,
        dest='mesh_resolution_y',
        help='Mesh resolution in y direction (default: %(default)s) - only applies to `rectangle` domain')
    parser.add_argument(
        '--mesh-resolution-sea-top-y',
        type=int,
        dest='mesh_resolution_sea_y',
        help='Mesh resolution for sea top beside ice shelf in y direction (default: %(default)s) - only applies to `rectangle` domain')
    parser.add_argument(
        '--store-sol',
        dest='store_solutions',
        action='store_true',
        help='Whether to save iteration solutions for display in Paraview')
    parser.add_argument(
        '--no-store-sol',
        dest='store_solutions',
        action='store_false',
        help='Whether to not save iteration solutions for display in Paraview')
    parser.set_defaults(store_solutions=None)
    parser.add_argument(
        '--label',
        help='Label to append to plots folder (default: %(default)s)')
    parser.add_argument(
        '--max-iter',
        type=int,
        dest='max_iter',
        help='Stop simulation after given number of timesteps; 0 = infinite (default: %(default)s)')
    parser.add_argument(
        '--non-linear-max-iter',
        type=int,
        dest='non_linear_max_iter',
        help='Stop non-linear Navier-Stokes loop given number of iterations (default: %(default)s)')
    parser.add_argument(
        '--plot-path',
        dest='plot_path',
        help='change Output folder (default is in "plots" based on timestamp)')
    parser.add_argument(
        '--stab',
        dest='stabilization',
        action='store_true',
        help='Whether to turn on stabilization method in simulation.')
    parser.add_argument(
        '--no-stab',
        dest='stabilization',
        action='store_false',
        help='Whether to not turn o nstabilization method in simulation.')
    parser.set_defaults(stabilization=None)
    parser.add_argument(
        '--delta0',
        type=float,
        dest='delta0',
        help='Value of stabilization coefficient delta0.')
    parser.add_argument(
        '--tau0',
        type=float,
        dest='tau0',
        help='Value of stabilization coefficient tau0.')
    parser.add_argument(
        '--degree-V',
        type=int,
        dest='degree_V',
        help='Velocity function space degree.')
    parser.add_argument(
        '--degree-P',
        type=int,
        dest='degree_P',
        help='Pressure function space degree.')
    parser.add_argument(
        '--degree-T',
        type=int,
        dest='degree_T',
        help='Temperature function space degree.')
    parser.add_argument(
        '--degree-S',
        type=int,
        dest='degree_S',
        help='Salinity function space degree.')
    parser.add_argument(
        '-vv, --very-verbose',
        dest='very_verbose',
        action='store_true',
        help='Whether to display debug information.')
    parser.add_argument(
        '--ms-to-kmh',
        dest='convert_from_ms_to_kmh',
        action='store_true',
        help='Whether to convert input constants (nu, g, rho_0) from m/s to km/h.')
    parser.set_defaults(convert_from_ms_to_kmh=None)

    commandline_args = parser.parse_args()
    commandline_args_dict = {arg: getattr(
        commandline_args, arg) for arg in vars(commandline_args)}

    # Purge None values before we send them for merge, otherwise they
    # would overwrite legit previous config values
    purged_commandline_args_dict = {
        key: val for key, val in commandline_args_dict.items()
        if val is not None}

    define_parameters(purged_commandline_args_dict)


def reload_status(plot_path):
    """Reload previous FEFFI status.
    Will read from a previous plot path and restore config, mesh and functions.
    After reloading status, one can jump straight to simulation.

    Parameters
    ----------
    plot_path : (str) directory containing a `config.yml` file and a sub-directory
                `solutions` with files `u.xml`, `p.xml`, `T.xml`, `S.xml`.

    Return
    ------
    f : (dict)
    domain : FEFFI Domain object
    mesh : FEniCS Mesh
    f_spaces : (dict)
    """

    global config

    # Load config
    config_file_path = os.path.join(plot_path, 'config.yml')
    with open(config_file_path) as saved_config:
        config = yaml.safe_load(saved_config)
        config['config_file'] = config_file_path

    # Load mesh, define function spaces and functions
    mesh = fenics.Mesh(os.path.join(plot_path, 'solutions', 'mesh.xml'))
    f_spaces = define_function_spaces(mesh)
    f = define_functions(f_spaces)
    domain = Domain(mesh, f_spaces)

    # Load functions
    fenics.File(os.path.join(plot_path, 'solutions', 'up.xml')) >> f['sol']
    fenics.File(os.path.join(plot_path, 'solutions', 'u.xml')) >> f['u_']
    fenics.File(os.path.join(plot_path, 'solutions', 'u.xml')) >> f['u_n']
    fenics.File(os.path.join(plot_path, 'solutions', 'p.xml')) >> f['p_']
    fenics.File(os.path.join(plot_path, 'solutions', 'p.xml')) >> f['p_n']
    fenics.File(os.path.join(plot_path, 'solutions', 'T.xml')) >> f['T_']
    fenics.File(os.path.join(plot_path, 'solutions', 'T.xml')) >> f['T_n']
    fenics.File(os.path.join(plot_path, 'solutions', 'S.xml')) >> f['S_']
    fenics.File(os.path.join(plot_path, 'solutions', 'S.xml')) >> f['S_n']

    return f, domain, mesh, f_spaces


def assemble_viscosity_tensor(visc):
    """Creates a proper viscosity tensor given relevant values.
    Notice that input must always be a list, even if it has only one element.

    Parameters
    ----------
    visc : list (of floats)
            If 1 entry is given, value will be used for all matrix entries.
            If 2 entries are given, they will be repeated row-wise.
            If 4 entries are given, they will compose the full tensor.

    Examples
    ----------
    1) Obtain a tensor of the form
       (a  a
        a  a)

       assemble_viscosity_tensor([a])

    2) Obtain a tensor of the form
       (a  b
        a  b)

       assemble_viscosity_tensor([a, b])

    3) Obtain a tensor of the form
       (a  b
        c  d)

       assemble_viscosity_tensor([a, b, c, d])
    """

    if len(visc) == 1:
        output = fenics.as_tensor((
            (fenics.Constant(visc[0]), fenics.Constant(visc[0])),
            (fenics.Constant(visc[0]), fenics.Constant(visc[0]))
        ))
    elif len(visc) == 2:
        output = fenics.as_tensor((
            (fenics.Constant(visc[0]), fenics.Constant(visc[1])),
            (fenics.Constant(visc[0]), fenics.Constant(visc[1]))
        ))
    elif len(visc) == 4:
        output = fenics.as_tensor((
            (fenics.Constant(visc[0]), fenics.Constant(visc[1])),
            (fenics.Constant(visc[2]), fenics.Constant(visc[3]))
        ))
    else:
        raise ValueError(
            "Viscosity needs 1, 2 or 4 entries input, %d given" % len(visc))

    return output


def convert_from_ms_to_kmh_input(config):
    """Rescale constants from m/s to km/h."""

    config['g'] *= 3.6*3.6*1000
    config['rho_0'] /= 3.6**2
    config['nu'] = [i*0.0036 for i in config['nu']]
    config['alpha'] = [i*0.0036 for i in config['alpha']]


def convert_from_ms_to_kmh_output(f):
    """
    """

    f['u_'] /= 3.6
