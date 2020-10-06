from . import parameters
from . import mesh
from . import functions
from . import boundaries
from . import simulation
from . import plot
import logging

logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('FFC').setLevel(logging.WARNING)

parameters.define_parameters()