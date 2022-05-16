import logging
from . import parameters
from . import mesh
from . import functions
from . import boundaries
from . import simulation
from . import plot
from . import meltparametrization
import fenics

fenics.parameters['allow_extrapolation'] = True

flog = logging.getLogger('feffii')

# Set up default config
parameters.define_parameters()
