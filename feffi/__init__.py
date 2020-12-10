import logging
from . import parameters
from . import mesh
from . import functions
from . import boundaries
from . import simulation
from . import plot

# Instantiate feffi logger
flog = logging.getLogger('feffi')
flog.setLevel(logging.INFO)
flog.propagate=False # dark magic https://stackoverflow.com/a/44426266

# Create two file handlers:
# one for file logging, another for terminal logging
#formatter = logging.Formatter('%(message)s')

fh = logging.FileHandler(
    'simulation.log',
    mode='w',
    encoding='utf-8')
fh.setLevel(logging.INFO)
#fh.setFormatter(formatter)
logging.getLogger('feffi').addHandler(fh)

th = logging.StreamHandler()
th.setLevel(logging.INFO)
#th.setFormatter(formatter)
logging.getLogger('feffi').addHandler(th)

# Reduce FEniCS logging to WARNING only
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('FFC').setLevel(logging.WARNING)

# Set up default config
parameters.define_parameters()