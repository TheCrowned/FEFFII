from fenics import * # assemble, File, solve, norm, XDMFFile
import fenics
from math import log
from pathlib import Path
from time import time
from datetime import datetime
from sys import exit
import os
import logging
import signal
import numpy as np
from . import parameters, plot
from .functions import build_NS_GLS_steady_form, build_temperature_form
flog = logging.getLogger('feffi')

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
        self.n = 0
        self.iterations_n = self.config['steps_n']*int(self.config['final_time'])
        #self.rounded_iterations_n = pow(10, (round(log(self.config['steps_n']*int(self.config['final_time']), 10))))

        if self.config['store_solutions']:
            self.xdmffile_sol = XDMFFile(os.path.join(self.config['plot_path'], 'solutions.xdmf'))
            self.xdmffile_sol.parameters["flush_output"] = True #https://github.com/FEniCS/dolfinx/issues/75
            self.xdmffile_sol.parameters["functions_share_mesh"] = True

            # Store mesh and first step solutions
            self.xdmffile_sol.write(self.f['u_'].function_space().mesh())
            self.save_solutions()

        flog.info('Initialized simulation')

    def run(self):
        """Runs the simulation until a stopping condition is met."""

        # Register SIGINT handler so we plot before exiting if CTRL-C is hit
        signal.signal(signal.SIGINT, self.sigint_handler)

        self.start_time = time()
        flog.info('Running full simulation; started at %s ' % str(datetime.now()))

        while self.n <= self.iterations_n:
            #plot.plot_solutions(self.f, display=True)
            self.timestep()

            if self.maybe_stop():
                self.log_progress()
                flog.info('Simulation ended at {}, after {} seconds.'.format(
                    str(datetime.now()), round(time()-self.start_time)))
                break

    def timestep(self):
        """Runs one timestep."""

        # ---------------------
        # Velocity and pressure
        # ---------------------

        # Init some vars
        V = self.f['sol'].split()[0].function_space()
        P = self.f['sol'].split()[1].function_space()
        mesh = V.mesh()
        (l, k) = (V.num_sub_spaces(), P.num_sub_spaces()+1) # f spaces degrees
        n = 0; residual_u = 1e22
        (self.f['u_n'], self.f['p_n']) = self.f['sol'].split(True)
        step_size = 1/self.config['steps_n']
        nu = self.config['nu']
        hmin = mesh.hmin(); hmax = mesh.hmax()
        tol = 10**self.config['simulation_precision']
        delta0 = self.config['delta0'] #1 # "tuning parameter" > 0
        tau0 = self.config['tau0'] #35 if l == 1 else 0 # "tuning parameter" > 0 dependent on V.degree

        # Solve the non linearity
        while residual_u > tol and n <= self.config['non_linear_max_iter']:

            a = self.f['sol'].split(True)[0]

            # ------------------------
            # Setting stab. parameters
            # ------------------------

            #norm_a = fenics.norm(a)
            norm_a = 1;
            if norm_a == 0: #first iteration, a = 0 -> would div by zero
               norm_a = 1

            Rej = norm_a*hmin/(2*nu[0])
            delta = delta0*hmin*min(1, Rej/3)/norm_a
            tau = tau0*max(nu[0], hmin)

            print('n = {}; Rej = {}; delta = {}; tau = {}'.format(
                n, round(Rej, 5), round(delta, 5), round(tau, 5)))

            u = self.f['u']; p = self.f['p']
            u_n = self.f['u_n'];
            v = self.f['v']; q = self.f['q']
            T_n = self.f['T_n']; S_n = self.f['S_n']

            # Define and solve NS problem
            steady_form = build_NS_GLS_steady_form(a, u, u_n, p, v, q, delta, tau, T_n, S_n)
            solve(fenics.lhs(steady_form) == fenics.rhs(steady_form),
                  self.f['sol'], bcs=self.BCs['V']+self.BCs['Q'])

            (self.f['u_'], self.f['p_']) = self.f['sol'].split(True)
            residual_u = fenics.errornorm(self.f['u_'], a)
            print(" >>> residual u: {}<<<\n". format(residual_u))

            n += 1

        # ------------------------
        # Temperature and salinity
        # ------------------------
        # Reassemble stiffness matrix and re-set BC, same for load vector, as coefficients change due to u_

        if self.config['beta'] != 0: #do not run if not coupled with velocity
            T_n = self.f['T_n']; T_v = self.f['T_v']; T = self.f['T']; u_ = self.f['u_']
            T_form = build_temperature_form(T, T_n, T_v, u_)
            solve(fenics.lhs(T_form) == fenics.rhs(T_form),
                  self.f['T_'], bcs=self.BCs['T'])

        '''if self.config['gamma'] != 0: #do not run if not coupled with velocity
            b5 = assemble(self.load_vectors['L5'])
            [bc.apply(b5) for bc in self.BCs['S']]
            A5 = assemble(self.stiffness_mats['a5'])
            [bc.apply(A5) for bc in self.BCs['S']]
            solve(A5, self.f['S_'].vector(), b5)'''


        self.errors = {
            'u' : fenics.errornorm(self.f['u_'], self.f['u_n']),
            'p' : fenics.errornorm(self.f['p_'], self.f['p_n']),
            'T' : fenics.errornorm(self.f['T_'], self.f['T_n']),
            'S' : fenics.errornorm(self.f['S_'], self.f['S_n']),
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
            self.save_solutions()

        # Prepare next timestep
        self.n = self.n + 1
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
            flog.info('Stopping simulation at step %d: all variables reached desired precision' % self.n)
            self.log_progress()
            return True

        if self.config['max_iter'] > 0 and self.n >= self.config['max_iter']:
            flog.info('Max iterations reached, stopping simulation at timestep %d' % self.n)
            return True

        if norm(self.f['u_'], 'L2') != norm(self.f['u_'], 'L2'):
            flog.info('Stopping simulation at step %d: velocity is NaN!' % self.n)
            return True

        return False

    def log_progress(self):
        """Logs simulation current timestep progress"""

        round_precision = abs(self.config['simulation_precision']) if self.config['simulation_precision'] <= 0 else 0

        flog.info('Timestep {} of {}:'. format(self.n, self.iterations_n))
        flog.info('  ||u|| = {}, ||u||_8 = {}, ||u-u_n|| = {}, ||u-u_n||/||u|| = {}'.format(round(norm(self.f['u_'], 'L2'), round_precision), round(norm(self.f['u_'].vector(), 'linf'), round_precision), round(self.errors['u'], round_precision), round(self.errors['u']/norm(self.f['u_'], 'L2'), round_precision)))
        flog.info('  ||p|| = {}, ||p||_8 = {}, ||p-p_n|| = {}, ||p-p_n||/||p|| = {}'.format(round(norm(self.f['p_'], 'L2'), round_precision), round(norm(self.f['p_'].vector(), 'linf'), round_precision), round(self.errors['p'], round_precision), round(self.errors['p']/norm(self.f['p_'], 'L2'), round_precision)))
        if self.config['beta'] > 0:
            flog.info('  ||T|| = {}, ||T||_8 = {}, ||T-T_n|| = {}, ||T-T_n||/||T|| = {}'.format(round(norm(self.f['T_'], 'L2'), round_precision), round(norm(self.f['T_'].vector(), 'linf'), round_precision), round(self.errors['T'], round_precision), round(self.errors['T']/norm(self.f['T_'], 'L2'), round_precision)))
        #if self.config['gamma'] > 0:
        #    flog.info('  ||S|| = {}, ||S||_8 = {}, ||S-S_n|| = {}, ||S-S_n||/||S|| = {}'.format(round(norm(self.f['S_'], 'L2'), round_precision), round(norm(self.f['S_'].vector(), 'linf'), round_precision), round(self.errors['S'], round_precision), round(self.errors['S']/norm(self.f['S_'], 'L2'), round_precision)))

    def save_solutions(self):
        """Saves current timestep solutions to XDMF file"""

        # t = dt*n
        t = self.n*self.config['final_time']*self.config['steps_n']

        self.xdmffile_sol.write(self.f['u_'], t)
        self.xdmffile_sol.write(self.f['p_'], t)
        self.xdmffile_sol.write(self.f['T_'], t)
        self.xdmffile_sol.write(self.f['S_'], t)

    def sigint_handler(self, sig, frame):
        """Catches CTRL-C when Simulation.run() is going,
        and plot solutions before exiting."""

        flog.info('Simulation stopped at {}, after {} seconds.\n'
                  'Jumping to plotting before exiting.'.format(
                    str(datetime.now()), round(time()-self.start_time)))
        plot.plot_solutions(self.f)
        os.system('xdg-open "' + self.config['plot_path'] + '"')
        exit(0)
