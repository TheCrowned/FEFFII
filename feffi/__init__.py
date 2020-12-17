import logging
from . import parameters
from . import mesh
from . import functions
from . import boundaries
from . import simulation
from . import plot

flog = logging.getLogger('feffi')

# Set up default config
parameters.define_parameters()