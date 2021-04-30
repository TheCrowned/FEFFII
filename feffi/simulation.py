from fenics import (assemble, File, solve, norm, XDMFFile, lhs, rhs, errornorm,
                    TestFunction, TrialFunction, Function, DirichletBC, dx,
                    Constant, VectorFunctionSpace, FunctionSpace, interpolate,
                    Expression)
from math import log
from pathlib import Path
from time import time
from datetime import datetime
from sys import exit
import os
import logging
import signal
import numpy as np
import yaml
from . import parameters, plot
from .functions import build_NS_GLS_steady_form, build_temperature_form, build_salinity_form
flog = logging.getLogger('feffi')


class Simulation(object):
    """Initializes a FEFFI model simulation.

    Parameters
    ----------
    f : dict
        Fenics functions used in variational formulations.
    BCs : dict
        Boundary conditions to be applied, as found in
        feffi.boundaries.Domain.BCs.

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
        domain = feffi.boundaries.Domain(
            mesh,
            f_spaces,
            domain='square'
        )
        simulation = feffi.simulation.Simulation(f, domain.BCs)
        simulation.run()
        feffi.plot.plot_solutions(f, display = True)
    """

    def __init__(self, f, BCs):
        self.f = f
        self.BCs = BCs
        self.n = 0
        self.iterations_n = parameters.config['steps_n']*int(parameters.config['final_time'])
        self.relative_errors = {}

        if parameters.config['store_solutions']:
            self.xdmffile_sol = XDMFFile(os.path.join(
                parameters.config['plot_path'], 'solutions.xdmf'))
            self.xdmffile_sol.parameters["flush_output"] = True #https://github.com/FEniCS/dolfinx/issues/75
            self.xdmffile_sol.parameters["functions_share_mesh"] = True

            # Store mesh and first step solutions
            #self.xdmffile_sol.write(self.f['u_'].function_space().mesh())
            self.save_solutions_xdmf()

        flog.info('Initialized simulation.')
        flog.info('Running parameters:\n' + str(parameters.config))

    def run(self):
        """Runs the simulation until a stopping condition is met."""

        # Register SIGINT handler so we plot before exiting if CTRL-C is hit
        signal.signal(signal.SIGINT, self.sigint_handler)

        self.start_time = time()
        flog.info('Running full simulation; started at {}'
                  .format(str(datetime.now())))

        while self.n <= self.iterations_n:
            self.timestep()

            if self.maybe_stop():
                self.log_progress()
                self.save_solutions_final()
                self.save_config()
                flog.info('Simulation stopped at {}, after {} steps ({} seconds).'.format(
                    str(datetime.now()), self.n, round(time()-self.start_time)))
                break

    def timestep(self):
        """Runs one timestep."""

        # ---------------------
        # Velocity and pressure
        # ---------------------

        # Init some vars
        self.nonlin_n = 0; residual_u = 1e22
        tol = 10**parameters.config['simulation_precision']
        g = Constant(parameters.config['g'])
        beta = Constant(parameters.config['beta'])
        T_0 = Constant(parameters.config['T_0'])
        rho_0 = Constant(parameters.config['rho_0'])
        pnh = Function(self.f['p_'].function_space()) #non-hydrostatic pressure

        # Obtain dph/dx
        # A linear space is used even though dT/dx will most likely be
        # piecewise constant.
        flog.debug('Solving for dph/dx...')
        dp_f_space = self.f['p_'].function_space()
        dph_dx = TrialFunction(dp_f_space)
        q = TestFunction(dp_f_space)
        dph_dx_sol = Function(dp_f_space)
        a = dph_dx.dx(1) * q * dx
        L = g*beta*(self.f['T_'].dx(0)) * q * dx
        bc = DirichletBC(dp_f_space, 0, 'near(x[1], 1)')
        solve(a == L, dph_dx_sol, bcs=[bc])
        flog.debug('Solved for dph/dx.')

        flog.debug('Interpolating dph/dx over 2D grid, with dph/dz=0...')
        grad_p_h = interpolate(Expression(('dph_dx', 0), dph_dx=dph_dx_sol,
                               degree=2), VectorFunctionSpace(dp_f_space.mesh(), 'Lagrange', 1))
        flog.debug('Interpolated dph/dx over 2D grid (norm = {}).'.format(norm(grad_p_h)))

        # Solve the non linearity
        flog.debug('Iteratively solving non-linear problem')
        while residual_u > tol and self.nonlin_n <= parameters.config['non_linear_max_iter']:
            a = self.f['sol'].split(True)[0] #this is the "u_n" of this non-linear loop

            # Shorthand for variables
            u = self.f['u']; p = self.f['p']
            u_n = self.f['u_n'];
            v = self.f['v']; q = self.f['q']
            T_n = self.f['T_n']; S_n = self.f['S_n']

            # Define and solve NS problem
            flog.debug('Solving for u,p...')
            steady_form = build_NS_GLS_steady_form(a, u, u_n, p, grad_p_h, v,
                                                   q, T_n, S_n)
            solve(lhs(steady_form) == rhs(steady_form), self.f['sol'],
                  bcs=self.BCs['V']+self.BCs['Q'])
            flog.debug('Solved for u,p.')

            (self.f['u_'], pnh) = self.f['sol'].split(True)
            residual_u = errornorm(self.f['u_'], a)
            flog.debug(" >>> residual u: {} <<<\n".format(residual_u))

            self.nonlin_n += 1
        flog.debug('Solved non-linear problem.')

        # Calculate ph and build full pressure as ph+pnh
        flog.debug('Solving for ph...')
        p_f_space = pnh.function_space()  # so we can avoid projecting later
        ph = TrialFunction(p_f_space)
        q = TestFunction(p_f_space)
        ph_sol = Function(p_f_space)
        a = ph.dx(1)/rho_0 * q * dx
        L = -g * (1 - beta*(self.f['T_']-T_0)) * q * dx
        bc = DirichletBC(dp_f_space, 0, 'near(x[1], 1)')
        solve(a == L, ph_sol, bcs=[bc])
        self.f['p_'].assign(pnh + ph_sol)
        flog.debug('Solved for ph.')

        # ------------------------
        # Temperature and salinity
        # ------------------------

        flog.debug('Solving for T and S.')

        if parameters.config['beta'] != 0: #do not run if not coupled with velocity
            T_form = build_temperature_form(self.f['T'], self.f['T_n'],
                                            self.f['T_v'], self.f['u_'])
            solve(lhs(T_form) == rhs(T_form), self.f['T_'], bcs=self.BCs['T'])

        if parameters.config['gamma'] != 0: #do not run if not coupled with velocity
            S_form = build_salinity_form(self.f['S'], self.f['S_n'],
                                         self.f['S_v'], self.f['u_'])
            solve(lhs(S_form) == rhs(S_form), self.f['S_'], bcs=self.BCs['S'])

        flog.debug('Solved for T and S.')
        self.relative_errors['u'] = errornorm(self.f['u_'], self.f['u_n'])/norm(self.f['u_'], 'L2') if norm(self.f['u_'], 'L2') != 0 else 0
        self.relative_errors['p'] = errornorm(self.f['p_'], self.f['p_n'])/norm(self.f['p_'], 'L2') if norm(self.f['p_'], 'L2') != 0 else 0
        self.relative_errors['T'] = errornorm(self.f['T_'], self.f['T_n'])/norm(self.f['T_'], 'L2') if norm(self.f['T_'], 'L2') != 0 else 0
        self.relative_errors['S'] = errornorm(self.f['S_'], self.f['S_n'])/norm(self.f['S_'], 'L2') if norm(self.f['S_'], 'L2') != 0 else 0

        self.log_progress()

        if parameters.config['store_solutions']:
            self.save_solutions_xdmf()

        # Prepare next timestep
        self.n += 1
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

        convergence_threshold = 10**(parameters.config['simulation_precision'])
        if all(error < convergence_threshold for error in self.relative_errors.values()):
            flog.info('Stopping simulation: all variables reached desired precision.')
            self.log_progress()
            return True

        if parameters.config['max_iter'] > 0 and self.n >= parameters.config['max_iter']:
            flog.info('Stopping simulation: max iterations reached.')
            return True

        if norm(self.f['u_'], 'L2') != norm(self.f['u_'], 'L2'):
            flog.error('Stopping simulation: velocity is NaN!')
            return True

        return False

    def log_progress(self):
        """Logs simulation current timestep progress"""

        round_precision = abs(parameters.config['simulation_precision']) if parameters.config['simulation_precision'] <= 0 else 0

        self.log('Timestep {} of {}:'. format(self.n, self.iterations_n))
        self.log('  Non-linearity u-P solved in {} steps.'.format(self.nonlin_n))
        self.log('  ||u|| = {}, ||u||_8 = {}, ||u-u_n|| = {}, ||u-u_n||/||u|| = {}'.format(
            round(norm(self.f['u_'], 'L2'), round_precision),
            round(norm(self.f['u_'].vector(), 'linf'), round_precision),
            round(self.relative_errors['u']*norm(self.f['u_'], 'L2'), round_precision),
            round(self.relative_errors['u'], round_precision)))
        self.log('  ||p|| = {}, ||p||_8 = {}, ||p-p_n|| = {}, ||p-p_n||/||p|| = {}'.format(
            round(norm(self.f['p_'], 'L2'), round_precision),
            round(norm(self.f['p_'].vector(), 'linf'), round_precision),
            round(self.relative_errors['p']*norm(self.f['p_'], 'L2'), round_precision),
            round(self.relative_errors['p'], round_precision)))
        if parameters.config['beta'] > 0: #avoid division by zero in relative error
            self.log('  ||T|| = {}, ||T||_8 = {}, ||T-T_n|| = {}, ||T-T_n||/||T|| = {}'.format(
                round(norm(self.f['T_'], 'L2'), round_precision),
                round(norm(self.f['T_'].vector(), 'linf'), round_precision),
                round(self.relative_errors['T']*norm(self.f['T_'], 'L2'), round_precision),
                round(self.relative_errors['T'], round_precision)))
        if parameters.config['gamma'] > 0:
            self.log('  ||S|| = {}, ||S||_8 = {}, ||S-S_n|| = {}, ||S-S_n||/||S|| = {}'.format(
                round(norm(self.f['S_'], 'L2'), round_precision),
                round(norm(self.f['S_'].vector(), 'linf'), round_precision),
                round(self.relative_errors['S']*norm(self.f['S_'], 'L2'), round_precision),
                round(self.relative_errors['S'], round_precision)))

    def save_solutions_xdmf(self):
        """Saves current timestep solutions to XDMF file (Paraview)"""

        #t = self.n*self.config['final_time']/self.config['steps_n']

        self.xdmffile_sol.write(self.f['u_'], self.n)
        self.xdmffile_sol.write(self.f['p_'], self.n)
        self.xdmffile_sol.write(self.f['T_'], self.n)
        self.xdmffile_sol.write(self.f['S_'], self.n)

    def save_solutions_final(self):
        """Saves last timestep solutions to XML files for later reusage in FEniCS"""

        (Path(os.path.join(parameters.config['plot_path'], 'solutions'))
            .mkdir(parents=True, exist_ok=True))

        (File(os.path.join(parameters.config['plot_path'], 'solutions', 'mesh.xml'))
            << self.f['u_'].function_space().mesh())
        (File(os.path.join(parameters.config['plot_path'], 'solutions', 'up.xml'))
            << self.f['sol'])
        (File(os.path.join(parameters.config['plot_path'], 'solutions', 'T.xml'))
            << self.f['T_'])
        (File(os.path.join(parameters.config['plot_path'], 'solutions', 'S.xml'))
            << self.f['S_'])

    def save_config(self):
        """Stores config used for simulation to file"""

        yaml.dump(parameters.config,
                  open(os.path.join(parameters.config['plot_path'], 'config.yml'), 'w'))

    def sigint_handler(self, sig, frame):
        """Catches CTRL-C when Simulation.run() is going,
        and plot solutions before exiting."""

        flog.info('Simulation stopped at {}, after {} steps ({} seconds).\n'
                  'Jumping to plotting before exiting.'.format(
                    str(datetime.now()), self.n, round(time()-self.start_time)))
        self.save_solutions_final()
        self.save_config()
        plot.plot_solutions(self.f)
        os.system('xdg-open "' + parameters.config['plot_path'] + '"')
        exit(0)

    def log(self, message):
        if self.n % 10 == 0 or parameters.config['very_verbose']:
            flog.info(message)
