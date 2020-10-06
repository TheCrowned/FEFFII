from fenics import assemble, File, solve, norm
from math import log
from pathlib import Path
from time import time
from datetime import datetime
from sys import exit
from os import system
import logging, signal
import numpy as np
from . import parameters, plot

class Simulation(object):
    """Initializes a FEFFI model simulation.

    Parameters
    ----------
    f : dict
        Fenics functions used in variational formulations.
    stiffness_mats : dict
        Stiffness matrices derived from variational forms.
    load_vectors: dict
        Load vectors derived from variational forms
    BCs : dict
        Boundary conditions to be applied, as found in
        feffi.boundaries.Domain.BCs.

    kwargs
    ------
    `final_time`, `steps_n`, `store_solutions`, `plot_path`, `simulation_precision`, `max_iter`.

    Attributes
    ----------
    config : dict
        FEFFI config
    errors : dict
        current timestamp solutions errors
    pvds : dict
        file handlers to pvds files for solution storage

    Examples
    --------
    1) Simulation over a square, with ending plots
        mesh = feffi.mesh.create_mesh(domain='square')
        f_spaces = feffi.functions.define_function_spaces()
        f = feffi.functions.define_functions(f_spaces)
        (stiffness_mats, load_vectors) = feffi.functions.define_variational_problems(f, mesh)
        domain = feffi.boundaries.Domain(
            mesh,
            f_spaces,
            domain='square'
        )
        simulation = feffi.simulation.Simulation(f, stiffness_mats, load_vectors, domain.BCs)
        simulation.run()
        feffi.plot.plot_solutions(f, display = True)
    """

    def __init__(self, f, stiffness_mats, load_vectors, BCs, **kwargs):
        # Allow function arguments to overwrite wide config (but keep it local)
        self.config = dict(parameters.config); self.config.update(kwargs)

        self.f = f
        self.stiffness_mats = stiffness_mats
        self.load_vectors = load_vectors
        self.BCs = BCs

        # Assemble stiffness matrices (a4, a5 need to be assembled at every time step)
        # Load vectors have coefficients which change upon every iteration
        self.A1 = assemble(self.stiffness_mats['a1'])
        self.A2 = assemble(self.stiffness_mats['a2'])
        self.A3 = assemble(self.stiffness_mats['a3'])

        # Apply boundary conditions
        [bc.apply(self.A1) for bc in BCs['V']]
        [bc.apply(self.A2) for bc in BCs['Q']]

        self.iterations_n = self.config['steps_n']*int(self.config['final_time'])
        self.rounded_iterations_n = pow(10, (round(log(self.config['steps_n']*int(self.config['final_time']), 10))))

        if self.config['store_solutions']:
            Path(self.config['plot_path'] + 'paraview/').mkdir(parents=True, exist_ok=True)
            self.pvds = {
                'u' : File(self.config['plot_path'] + 'paraview/velocity.pvd'),
                'p' : File(self.config['plot_path'] + 'paraview/pressure.pvd'),
                'T' : File(self.config['plot_path'] + 'paraview/temperature.pvd'),
                'S' : File(self.config['plot_path'] + 'paraview/salinity.pvd')
            }

            self.save_pvds()

        logging.info('Initialized simulation')

    def run(self):
        """Runs the simulation until a stopping condition is met."""

        # Register SIGINT handler so we plot before exiting if CTRL-C is hit
        signal.signal(signal.SIGINT, self.sigint_handler)

        start_time = time()
        logging.info('Running full simulation; started at %s ' % str(datetime.now()))

        for self.n in range(self.iterations_n):
            self.timestep()
            start, last_run = 0, 0

            if self.maybe_stop():
                self.log_progress()
                break

    def timestep(self):
        """Runs one timestep."""

        # Applying IPC splitting scheme (IPCS)
        # Step 1: Tentative velocity step
        A1 = self.A1
        b1 = assemble(self.load_vectors['L1'])
        [bc.apply(b1) for bc in self.BCs['V']]
        solve(A1, self.f['u_'].vector(), b1)

        # Step 2: Pressure correction step
        A2 = self.A2
        b2 = assemble(self.load_vectors['L2'])
        [bc.apply(b2) for bc in self.BCs['Q']]
        solve(A2, self.f['p_'].vector(), b2)

        # Step 3: Velocity correction step
        A3 = self.A3
        b3 = assemble(self.load_vectors['L3'])
        solve(A3, self.f['u_'].vector(), b3)

        # Step 4: Temperature step
        # Reassemble stiffness matrix and re-set BC, same for load vector, as coefficients change due to u_
        b4 = assemble(self.load_vectors['L4'])
        [bc.apply(b4) for bc in self.BCs['T']]
        A4 = assemble(self.stiffness_mats['a4'])
        [bc.apply(A4) for bc in self.BCs['T']]
        solve(A4, self.f['T_'].vector(), b4)

        # Step 5: Salinity step
        # Reassemble stiffness matrix and re-set BC, same for load vector, as coefficients change due to u_
        b5 = assemble(self.load_vectors['L5'])
        [bc.apply(b5) for bc in self.BCs['S']]
        A5 = assemble(self.stiffness_mats['a5'])
        [bc.apply(A5) for bc in self.BCs['S']]
        solve(A5, self.f['S_'].vector(), b5)

        self.errors = {
            'u' : np.linalg.norm(self.f['u_'].vector().get_local() - self.f['u_n'].vector().get_local()),
            'p' : np.linalg.norm(self.f['p_'].vector().get_local() - self.f['p_n'].vector().get_local()),
            'T' : np.linalg.norm(self.f['T_'].vector().get_local() - self.f['T_n'].vector().get_local()),
            'S' : np.linalg.norm(self.f['S_'].vector().get_local() - self.f['S_n'].vector().get_local())
        }

        """
        # Even if verbose, get progressively less verbose with the order of number of iterations
        if(self.args.very_verbose or (rounded_iterations_n < 1000 or (rounded_iterations_n >= 1000 and n % (rounded_iterations_n/100) == 0))):
        last_run = time.time() - start
        #last_run = (last_run*(n/100 - 1) + (time.time() - start))/(n/100) if n != 0 else 0
        eta = round(last_run*(iterations_n - n))/100 if last_run != 0 else '?'
        start = time.time()

        self.log_progress()
        """

        self.log_progress()

        if self.config['store_solutions']:
            self.save_pvds()

        # Prepare next timestep
        self.f['u_n'].assign(self.f['u_'])
        self.f['p_n'].assign(self.f['p_'])
        self.f['T_n'].assign(self.f['T_'])
        self.f['S_n'].assign(self.f['S_'])

    def maybe_stop(self):
        """Checks whether simulation should be stopped.

        Criteria are:
             - all variables have reached simulation precision
             - reached max number of iterations
             - simulation has diverged (i.e. velocity is NaN)

        Return
        ------
        bool : whether simulation should be stopped.
        """

        convergence_threshold = 10**(self.config['simulation_precision'])
        if all(error < convergence_threshold for error in self.errors.values()):
            logging.info('Stopping simulation at step %d: all variables reached desired precision' % self.n)
            self.log_progress()
            return True

        if self.config['max_iter'] > 0 and self.n >= self.config['max_iter']:
            logging.info('Max iterations reached, stopping simulation at timestep %d' % self.n)
            return True

        if norm(self.f['u_'], 'L2') != norm(self.f['u_'], 'L2'):
            logging.info('Stopping simulation at step %d: velocity is NaN!' % self.n)
            return True

        return False

    def log_progress(self):
        """Logs simulation current timestep progress"""

        round_precision = abs(self.config['simulation_precision']) if self.config['simulation_precision'] <= 0 else 0

        logging.info("Timestep %d of %d: \n  ||u|| = %s, ||u||_8 = %s, ||u-u_n|| = %s, ||u-u_n||/||u|| = %s, \n  ||p|| = %s, ||p||_8 = %s, ||p-p_n|| = %s, ||p-p_n||/||p|| = %s, \n  ||T|| = %s, ||T||_8 = %s, ||T-T_n|| = %s, ||T-T_n||/||T|| = %s, \n  ||S|| = %s, ||S||_8 = %s, ||S - S_n|| = %s, ||S - S_n||/||S|| = %s" % ( \
            self.n, self.iterations_n, \
            round(norm(self.f['u_'], 'L2'), round_precision), round(norm(self.f['u_'].vector(), 'linf'), round_precision), round(self.errors['u'], round_precision), round(self.errors['u']/norm(self.f['u_'], 'L2'), round_precision), \
            round(norm(self.f['p_'], 'L2'), round_precision), round(norm(self.f['p_'].vector(), 'linf'), round_precision), round(self.errors['p'], round_precision), round(self.errors['p']/norm(self.f['p_'], 'L2'), round_precision), \
            round(norm(self.f['T_'], 'L2'), round_precision), round(norm(self.f['T_'].vector(), 'linf'), round_precision), round(self.errors['T'], round_precision), round(self.errors['T']/norm(self.f['T_'], 'L2'), round_precision), \
            round(norm(self.f['S_'], 'L2'), round_precision), round(norm(self.f['S_'].vector(), 'linf'), round_precision), round(self.errors['S'], round_precision), round(self.errors['S']/norm(self.f['S_'], 'L2'), round_precision),
        ) )

    def save_pvds(self):
        """Saves current timestep solutions to pvd files"""

        self.pvds['u'] << self.f['u_']
        self.pvds['p'] << self.f['p_']
        self.pvds['T'] << self.f['T_']
        self.pvds['S'] << self.f['S_']

    def sigint_handler(self, sig, frame):
        logging.info('Simulation stopped -- jumping to plotting before exiting')
        plot.plot_solutions(self.f)
        system('xdg-open "' + self.config['plot_path'] + '"')
        exit(0)