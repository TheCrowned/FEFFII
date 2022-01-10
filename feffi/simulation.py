from fenics import (assemble, File, solve, norm, XDMFFile, lhs, rhs, dx, div,
                    TestFunction, TrialFunction, Function, DirichletBC,
                    Constant, VectorFunctionSpace, FunctionSpace,
                    interpolate, Expression, project)
import math
from pathlib import Path
from time import time
from datetime import datetime
from sys import exit
import os
import logging
import signal
import numpy as np
import yaml
#import csv
from . import parameters, plot
from .parameters import convert_constants_from_kmh_to_ms
from .functions import (build_NS_GLS_steady_form, build_temperature_form,
                        build_salinity_form, build_buoyancy, energy_norm)
from .meltparametrization import solve_3eqs_system, uStar_expr
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

    def __init__(self, f, domain):
        self.f = f
        self.domain = domain
        self.BCs = domain.BCs
        self.n = 0
        self.iterations_n = parameters.config['steps_n']*int(parameters.config['final_time'])
        self.relative_errors = {}
        self.dim = self.domain.mesh.geometric_dimension() # 2D or 3D
        self.z_coord = self.dim-1 # z-coord in mesh points changes depending on 2/3D

        if parameters.config['simulation_precision'] <= 0:
            self.round_precision = abs(parameters.config['simulation_precision'])
        else:
            self.round_precision = 0

        if parameters.config['store_solutions']:
            self.xdmffile_sol = XDMFFile(os.path.join(
                parameters.config['plot_path'], 'solutions.xdmf'))
            self.xdmffile_sol.parameters["flush_output"] = True #https://github.com/FEniCS/dolfinx/issues/75
            self.xdmffile_sol.parameters["functions_share_mesh"] = True

            # Store mesh and first step solutions
            #self.xdmffile_sol.write(self.f['u_'].function_space().mesh())
            self.save_solutions_xdmf()

        '''csv_simul_data_file = open(
            os.path.join(parameters.config['plot_path'], 'simul_data.csv'),
                         'w')
        fieldnames = ['n',
                      '||u||_2', '||u||_inf', 'E(u)',
                      '||p||_2', '||p||_inf', 'E(p)',
                      '||T||_2', '||T||_inf', 'E(T)',
                      '||S||_2', '||S||_inf', 'E(S)']
        self.csv_simul_data = csv.DictWriter(
            csv_simul_data_file, delimiter=',', quotechar='"',
            quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
        self.csv_simul_data.writeheader()'''

        flog.info('Initialized simulation.')
        flog.info('Running parameters:\n' + str(parameters.config))
        flog.info('Mesh vertices {}, hmin {}, hmax {}'.format(
            domain.mesh.num_vertices(),
            round(domain.mesh.hmin(), 4), round(domain.mesh.hmax(), 4)))

    def run(self):
        """Runs the simulation until a stopping condition is met."""

        # Register SIGINT handler so we plot before exiting if CTRL-C is hit
        signal.signal(signal.SIGINT, self.sigint_handler)

        self.start_time = time()
        flog.info('Running full simulation; started at {}'
                  .format(str(datetime.now())))

        while self.n <= self.iterations_n:
            self.timestep()
            self.log_progress()

            # Plot/Save solutions every given iterations so we can keep an eye
            if parameters.config['checkpoint_interval'] != 0:
                if self.n > 0 and self.n % parameters.config['checkpoint_interval'] == 0:
                    flog.debug('--- Save Checkpoint at Timestep {} ---'.format(self.n))
                    plot.plot_solutions(self.f)
                    self.save_solutions_final()

                    # Store solution for paraview
                    if parameters.config['store_solutions']:
                        self.save_solutions_xdmf()

            # If simulation is over
            if self.maybe_stop():
                flog.info('Simulation stopped at {}, after {} steps ({} seconds).'.format(
                    str(datetime.now()), self.n, round(time()-self.start_time)))
                break

        self.final_operations()

    def final_operations(self):
        """Performs ending-simulation tasks, such as plotting and solutions saving."""

        self.build_full_pressure(self.f['p_'])
        self.save_solutions_final()
        plot.plot_solutions(self.f)
        self.save_config()

        # Store solution for paraview
        if parameters.config['store_solutions']:
            self.save_solutions_xdmf()

    def timestep(self):
        """Runs one timestep."""

        # ---------------------
        # VELOCITY AND PRESSURE
        # ---------------------

        # Init some vars
        self.nonlin_n = 0; residual_u = 1e22
        tol = 10**parameters.config['simulation_precision']
        g = Constant(parameters.config['g'])
        beta = Constant(parameters.config['beta'])
        gamma = Constant(parameters.config['gamma'])
        rho_0 = Constant(parameters.config['rho_0'])
        pnh = Function(self.f['p_'].function_space()) # non-hydrostatic pressure

        '''
        # ----------------
        # Calculate dph/dx
        # ----------------
        # Use P function space even if we compute gradient, which should be one degree lower

        flog.debug('Solving for dph/dx...')
        dph_dx_f_space = self.f['p_'].function_space()
        dph_dx = TrialFunction(dph_dx_f_space)
        q = TestFunction(dph_dx_f_space)
        dph_dx_sol = Function(dph_dx_f_space)

        if parameters.config['pressure_split']:
            a = dph_dx.dx(self.z_coord) * q * dx
            L = -g * (-beta*self.f['T_'].dx(0)+gamma*self.f['S_'].dx(0)) * q * dx
            bc_domain = ('near(x[{}], {})'.format(self.z_coord, max(self.domain.mesh.coordinates()[:, self.z_coord])))
            bc = DirichletBC(dph_dx_f_space, 0, bc_domain)
            solve(a == L, dph_dx_sol, bcs=[bc])
        else:
            a = dph_dx * q * dx# + 0.1*dph_dx.dx(z_coord) * q.dx(z_coord)*dx
            L = -g * (-beta*self.f['T_']+gamma*self.f['S_']) * q * dx
            solve(a == L, dph_dx_sol)

        flog.debug('Solved for dph/dx.')

        if self.dim == 2:
            if parameters.config['pressure_split']:
                grad_ph_tup = ('dph_dx', 0)
            else:
                grad_ph_tup = (0, '-dph_dx') #switching sign since it will be added to lhs

        elif self.dim == 3:
            grad_ph_tup = ('dph_dx', 0, 0)

        grad_ph = interpolate(Expression(grad_ph_tup, dph_dx=dph_dx_sol, degree=2),
                              VectorFunctionSpace(dph_dx_f_space.mesh(), 'Lagrange', 1))
        # Linear space is used for grad_ph even though dT/dx will most likely be
        # piecewise constant. Can't hurt, I guess.

        flog.debug('Interpolated dph/dx over 2D grid (norm = {}).'.format(round(norm(grad_ph), 2)))
        '''

        # --------------------------
        # Solve GLS Navier-Stokes eq
        # --------------------------

        # Shorthand for variables
        u = self.f['u']; p = self.f['p']
        v = self.f['v']; q = self.f['q']
        u_n = self.f['u_n']; T_n = self.f['T_n']; S_n = self.f['S_n']

        # Define BCs
        bcs = []
        if self.BCs.get('V'):
            bcs += self.BCs['V']
        if self.BCs.get('Q'):
            bcs += self.BCs['Q']

        # Solve non-linearity iteratively
        flog.debug('Iteratively solving non-linear problem')
        while residual_u > tol and self.nonlin_n <= parameters.config['non_linear_max_iter']:
            a = self.f['sol'].split(True)[0] # this is the "u_n" of this non-linear loop

            # Define and solve NS problem
            flog.debug('Solving for u, p...')
            steady_form = build_NS_GLS_steady_form(a, u, u_n, p, 0, v,
                                                   q, T_n, S_n)
            solve(lhs(steady_form) == rhs(steady_form), self.f['sol'], bcs=bcs)
                  #solver_parameters={'linear_solver':'mumps'})
            flog.debug('Solved for u, p.')

            (self.f['u_'], pnh) = self.f['sol'].split(True) # only used to calculate residual
            #residual_u = norm(project(self.f['u_']-a, a.function_space()), 'L2')
            #print(residual_u)
            residual_u = np.linalg.norm(self.f['u_'].compute_vertex_values() - a.compute_vertex_values(), ord=2)

            flog.debug('>>> residual u: {} <<<'.format(residual_u))
            self.nonlin_n += 1

        flog.debug('Solved non-linear problem (u, p).')

        self.f['p_'].assign(pnh)

        # ------------------------
        # TEMPERATURE AND SALINITY
        # ------------------------

        # Solve 3 equations system to obtain T and S forcing terms
        if parameters.config['melt_boundaries'] != [None]:
            if self.n % 1 == 0:
                flog.debug('Solving 3 equations system...')

                # Compute uStar
                self.f['3eqs']['uStar'] = interpolate(
                    uStar_expr(self.f['u_']),
                    self.f['u_'].function_space()
                ).sub(0)

                solve_3eqs_system(self.f) # result goes into self.f['3eqs']

                # These log values are useless, we should only compute them at ice boundary
                flog.debug('Solved 3 equations system.')
                '''            ' - mw: {}\n - Tzd: {}\n - Szd: {}'
                            .format(round(np.average(mw.compute_vertex_values()), self.round_precision),
                                    round(np.average(Tzd.compute_vertex_values()), self.round_precision),
                                    round(np.average(Szd.compute_vertex_values()), self.round_precision))))'''

        # Other functions will check if m_B == False to determine whether melt
        # parametrization is enabled in this run
        else:
            self.f['3eqs']['m_B'], self.f['3eqs']['T_B'], self.f['3eqs']['S_B'] = False, False, False

        flog.debug('Solving for T and S...')

        '''BCs_T = self.BCs['T']
        BCs_T.append(0)
        T_B_boundary = Expression('T_B', degree=2, T_B=Tzd)
        BCs_T[-1] = DirichletBC(self.f['T_'].function_space(), T_B_boundary, self.domain.marked_subdomains, self.domain.subdomains_markers['left_ice'])

        BCs_S = self.BCs['S']
        BCs_S.append(0)
        S_B_boundary = Expression('S_B', degree=2, S_B=Szd)
        BCs_S[-1] = DirichletBC(self.f['S_'].function_space(), S_B_boundary, self.domain.marked_subdomains, self.domain.subdomains_markers['left_ice'])'''

        if parameters.config['beta'] != 0: #do not run if not coupled with velocity
            T_form = build_temperature_form(self.f, self.domain)
            solve(lhs(T_form) == rhs(T_form), self.f['T_'], bcs=self.BCs['T'])

        if parameters.config['gamma'] != 0: #do not run if not coupled with velocity
            S_form = build_salinity_form(self.f, self.domain)
            solve(lhs(S_form) == rhs(S_form), self.f['S_'], bcs=self.BCs['S'])

        flog.debug('Solved for T and S.')

        self.relative_errors['u'] = (np.linalg.norm(self.f['u_'].compute_vertex_values() - self.f['u_n'].compute_vertex_values()))/np.linalg.norm(self.f['u_'].compute_vertex_values()) if norm(self.f['u_'], 'L2') != 0 else 0
        self.relative_errors['p'] = (np.linalg.norm(self.f['p_'].compute_vertex_values() - self.f['p_n'].compute_vertex_values()))/np.linalg.norm(self.f['p_'].compute_vertex_values()) if norm(self.f['p_'], 'L2') != 0 else 0
        self.relative_errors['T'] = (np.linalg.norm(self.f['T_'].compute_vertex_values() - self.f['T_n'].compute_vertex_values()))/np.linalg.norm(self.f['T_'].compute_vertex_values()) if norm(self.f['T_'], 'L2') != 0 else 0
        self.relative_errors['S'] = (np.linalg.norm(self.f['S_'].compute_vertex_values() - self.f['S_n'].compute_vertex_values()))/np.linalg.norm(self.f['S_'].compute_vertex_values()) if norm(self.f['S_'], 'L2') != 0 else 0

        '''csv_row = {'n': self.n}
        for func in ['u', 'p', 'T', 'S']:
            csv_row.update({
                '||{}||_2'.format(func): norm(self.f[func+'_'], 'L2'),
                '||{}||_inf'.format(func): norm(self.f[func+'_'].vector(), 'linf'),
                'E({})'.format(func): energy_norm(self.f[func+'_'])
            })
        self.csv_simul_data.writerow(csv_row)'''

        #if self.n % 100 == 0:
            #if pressuresplit:
            #    plot.plot_single(dph_dx_sol,display=True,title='hydrostatic pressure gradient')
            #else:
            #    plot.plot_single(dph_dx_sol,display=True,title='delta rho g')

            #plot.plot_single(drho_dx_sol,display=True,title='rho gradient')


        # Prepare next timestep
        self.n += 1
        self.f['u_n'].assign(self.f['u_'])
        self.f['p_n'].assign(self.f['p_'])
        self.f['T_n'].assign(self.f['T_'])
        self.f['S_n'].assign(self.f['S_'])

    def build_full_pressure(self, pnh):
        """Build full pressure as sum of hydrostatic and non-hydrostatic components."""

        # ------------
        # Calculate ph
        # ------------
        if parameters.config['pressure_split']:
            flog.debug('Solving for ph...')
            ph_f_space = self.f['p_'].function_space()
            ph = TrialFunction(ph_f_space)
            q = TestFunction(ph_f_space)
            ph_sol = Function(ph_f_space)
            a = ph.dx(self.z_coord) * q * dx
            #a = ph.dx(self.z_coord)/rho_0 * q * dx
            L = build_buoyancy(self.f['T_'], self.f['S_']) * q * dx
            bc_domain = ('near(x[{}], {})'.format(self.z_coord, max(self.domain.mesh.coordinates()[:, self.z_coord])))
            bc = DirichletBC(ph_f_space, 0, bc_domain)
            solve(a == L, ph_sol, bcs=[bc])
            flog.debug('Solved for ph.')
            #plot.plot_single(pnh, display=True)
            #plot.plot_single(ph_sol, display=True)
            self.f['p_'].assign(pnh + ph_sol)

        self.f['p_'].assign(parameters.config['rho_0'] * self.f['p_'])


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

        # Calculate \int_\Omega \nabla \cdot u * dx
        try:
            div_u = math.floor(math.log10(abs(assemble(div(self.f['u_']) * dx))))
        except ValueError: # if 0, math.log10 will complain
            div_u = '-inf'

        self.log('Timestep {} of {}:'. format(self.n, self.iterations_n))
        self.log('  Non-linearity u-P solved in {} steps.'.format(self.nonlin_n))
        self.log('  avg(u) = ({}, {}), max(u) = ({}, {})'.format(
            round(np.average(self.f['u_'].sub(0).compute_vertex_values()), self.round_precision),
            round(np.average(self.f['u_'].sub(1).compute_vertex_values()), self.round_precision),
            round(max(self.f['u_'].sub(0).compute_vertex_values()), self.round_precision),
            round(max(self.f['u_'].sub(1).compute_vertex_values()), self.round_precision)))
        self.log('  ||u-u_n|| = {}, ||u-u_n||/||u|| = {}, div(u) = 1e{}'.format(
            round(self.relative_errors['u']*norm(self.f['u_'], 'L2'), self.round_precision),
            round(self.relative_errors['u'], self.round_precision),
            div_u))
        self.log('  avg(p) = {}, ||p||_8 = {}'.format(
            round(np.average(self.f['p_'].compute_vertex_values()), self.round_precision),
            round(norm(self.f['p_'].vector(), 'linf'), self.round_precision)))
        self.log('  ||p-p_n|| = {}, ||p-p_n||/||p|| = {}'.format(
            round(self.relative_errors['p']*norm(self.f['p_'], 'L2'), self.round_precision),
            round(self.relative_errors['p'], self.round_precision)))
        if parameters.config['beta'] > 0: #avoid division by zero in relative error
            self.log('  avg(T) = {}, ||T||_8 = {}'.format(
                round(np.average(self.f['T_'].compute_vertex_values()), self.round_precision),
                round(norm(self.f['T_'].vector(), 'linf'), self.round_precision)))
            self.log('  ||T-T_n|| = {}, ||T-T_n||/||T|| = {}'.format(
                round(self.relative_errors['T']*norm(self.f['T_'], 'L2'), self.round_precision),
                round(self.relative_errors['T'], self.round_precision)))
        if parameters.config['gamma'] > 0:
            self.log('  avg(S) = {}, ||S||_8 = {}'.format(
                round(np.average(self.f['S_'].compute_vertex_values()), self.round_precision),
                round(norm(self.f['S_'].vector(), 'linf'), self.round_precision)))
            self.log('  ||S-S_n|| = {}, ||S-S_n||/||S|| = {}'.format(
                round(self.relative_errors['S']*norm(self.f['S_'], 'L2'), self.round_precision),
                round(self.relative_errors['S'], self.round_precision)))

    def save_solutions_xdmf(self):
        """Saves current timestep solutions to XDMF file (Paraview)"""

        #t = self.n*self.config['final_time']/self.config['steps_n']

        self.xdmffile_sol.write(self.f['u_'].sub(0), self.n)
        self.xdmffile_sol.write(self.f['u_'].sub(1), self.n)
        self.xdmffile_sol.write(self.f['p_'], self.n)
        self.xdmffile_sol.write(self.f['T_'], self.n)
        self.xdmffile_sol.write(self.f['S_'], self.n)

    def save_solutions_final(self):
        """Saves last timestep solutions to XML files for later reusage in FEniCS"""

        (Path(os.path.join(parameters.config['plot_path'], 'solutions'))
            .mkdir(parents=True, exist_ok=True))

        (File(os.path.join(parameters.config['plot_path'], 'solutions', 'mesh.xml'))
            << self.f['u_'].function_space().mesh())
        (File(os.path.join(parameters.config['plot_path'], 'solutions', 'up_{}.xml'.format(self.n)))
            << self.f['sol'])
        (File(os.path.join(parameters.config['plot_path'], 'solutions', 'u_{}.xml'.format(self.n)))
            << self.f['u_'])
        (File(os.path.join(parameters.config['plot_path'], 'solutions', 'p_{}.xml'.format(self.n)))
            << self.f['p_'])
        (File(os.path.join(parameters.config['plot_path'], 'solutions', 'T_{}.xml'.format(self.n)))
            << self.f['T_'])
        (File(os.path.join(parameters.config['plot_path'], 'solutions', 'S_{}.xml'.format(self.n)))
            << self.f['S_'])

    def save_config(self):
        """Stores config used for simulation to file."""

        # If constants had been converted from m/s to km/h, revert them back
        if (parameters.config.get('convert_from_ms_to_kmh') and
            parameters.config['convert_from_ms_to_kmh']):
            to_save_config = convert_constants_from_kmh_to_ms(parameters.config)
        else:
            to_save_config = parameters.config

        with open(os.path.join(parameters.config['plot_path'],
                               'config.yml'), 'w') as save_handle:
            yaml.safe_dump(to_save_config, save_handle)

    def sigint_handler(self, sig, frame):
        """Catches CTRL-C when Simulation.run() is going,
        and plot solutions before exiting."""

        flog.info('Simulation stopped at {}, after {} steps ({} seconds).\n'
                  'Jumping to plotting before exiting.'.format(
                    str(datetime.now()), self.n, round(time()-self.start_time)))
        self.final_operations()
        os.system('xdg-open "' + parameters.config['plot_path'] + '"')
        exit(0)

    def log(self, message):
        if self.n % 10 == 0 or parameters.config['very_verbose']:
            flog.info(message)
