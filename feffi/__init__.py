from . import parameters
from . import mesh
from . import functions
from . import boundaries
from . import simulation
from . import plot
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('FFC').setLevel(logging.WARNING)

parameters.define_parameters()