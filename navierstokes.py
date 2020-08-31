"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for channel flow on the unit square using the
Incremental Pressure Correction Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
								 div(u) = 0

Copyright (C) 2020 Stefano Ottolenghi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import print_function
from datetime import datetime
from fenics import *
#from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import sys, getopt, argparse, pathlib
import math
import inspect
from importlib import import_module
import time, os
from shelfgeometry import ShelfGeometry
import yaml
import signal

class NavierStokes(object):

	def __init__(self):
		parameters['allow_extrapolation'] = True

		self.config = yaml.safe_load(open("config.yml"))
		#print(self.config)
		#args_dict = {arg: getattr(args, arg) for arg in vars(args)}

		self.args = self.parse_commandline_args()

		label = " --label " + self.args.label if self.args.label else ""
		self.plot_path = 'plots/%d --final-time %.0f --steps-n %d --mesh-resolution %d%s/' % (round(time.time()), self.args.final_time, self.args.steps_n, self.args.mesh_resolution, label)
		pathlib.Path(self.plot_path).mkdir(parents=True, exist_ok=True)
		self.log_file = open(self.plot_path + 'simulation.log', 'w')

		def assemble_viscosity():
			nu = [i*0.0036 for i in self.args.nu] # from m^2/s to km^2/h

			if len(nu) == 1:
				viscosity = as_tensor((
					(nu[0], 0),
					(0, nu[0])
				))

			elif len(self.args.nu) == 2:
				viscosity = as_tensor((
					(nu[0], 0),
					(0, nu[1])
				))

			elif len(self.args.nu) == 4:
				viscosity = as_tensor((
					(nu[0], nu[1]),
					(nu[2], nu[3])
				))

			else:
				raise ValueError("Viscosity needs 1, 2 or 4 entries input, %d given" % len(nu))

			return viscosity

		# Values used in variational forms, some provided as input from terminal
		# For units of measure, see README.md
		self.const = Bunch({
			'dt': Constant(1 / self.args.steps_n),
			'nu': assemble_viscosity(),
			'rho_0': Constant(self.args.rho_0/(3.6**2)), 	# from kg/m^3 to h^2*Pa/km^2
			'g': Constant(1.27*10**5),						# from m/s^2 to km/h^2
			'alpha': Constant(self.args.alpha),
			'beta': Constant(self.args.beta),
			'T_0': Constant(self.args.T_0),
			'S_0': Constant(self.args.S_0),
			'ocean_bc': Constant(self.args.ocean_bc)
		})

		self.round_precision = abs(self.args.simulation_precision) if self.args.simulation_precision <= 0 else 0

		# tolerance for near() function based on mesh resolution. Otherwise BC are not properly set
		# DO WE NEED THIS??
		tolerance = pow(10, - round(math.log(self.args.mesh_resolution, 10)))

		#If --very-verbose is requested, enable FEniCS debug mode
		#if(self.args.very_verbose == True):
			#set_log_active(True)
			#set_log_level(1)

		# Register SIGINT handler so we plot before exiting
		signal.signal(signal.SIGINT, self.sigint_handler)

		#Import correct set of boundaries depending on domain and set parameters
		self.bd = import_module('boundaries_' + self.args.domain)
		self.bd.args = self.args
		self.bd.tolerance = tolerance

		self.function_spaces = Bunch({})
		self.functions = Bunch({})
		self.stiffness_mats = Bunch({})
		self.load_vectors = Bunch({})
		self.bcu, self.bcp, self.bcT, self.bcS = [], [], [], []

		self.start_time = time.time()
		self.log('--- Started at %s --- ' % str(datetime.now()), True)
		self.log('--- Parameters are: ---', True)
		self.log(str(self.args), True)

	def parse_commandline_args(self):

		# From https://stackoverflow.com/questions/53937481/python-command-line-arguments-foo-and-no-foo
		def add_bool_arg(parser, name, default=False, dest=None, help=None):
			if(dest is None):
				dest = name

			group = parser.add_mutually_exclusive_group(required=False)
			group.add_argument('--' + name, dest=dest, action='store_true', help=help)
			group.add_argument('--no-' + name, dest=dest, action='store_false', help=help)
			parser.set_defaults(**{name:default})

		# See https://docs.python.org/2/library/argparse.html
		parser = argparse.ArgumentParser()
		parser.add_argument('--final-time', default=self.config['final_time'], type=float, dest='final_time', help='How long to run the simulation for (hours) (default: %(default)s)')
		parser.add_argument('--steps-n', default=self.config['steps_n'], type=int, dest='steps_n', help='How many steps each of the "seconds" is made of (default: %(default)s)')
		parser.add_argument('--precision', default=self.config['precision'], type=int, dest='simulation_precision', help='Precision at which converge is achieved, for all variables (power of ten) (default: %(default)s)')
		parser.add_argument('--viscosity', default=self.config['nu'], type=float, dest='nu', nargs="*", help='Viscosity, m^2/s. Expects 1, 2 or 4 space-separated entries, depending on whether a scalar, vector or tensor is wished (default: %(default)s)')
		parser.add_argument('--rho-0', default=self.config['rho_0'], type=float, dest='rho_0', help='Density, kg/m^3 (default: %(default)s)')
		parser.add_argument('--alpha', default=self.config['alpha'], type=float, help='Water thermal expansion coefficient, 1/°C (default: %(default)s)')
		parser.add_argument('--beta', default=self.config['beta'], type=float, help='Water salinity expansion coefficient, 1/PSU (default: %(default)s)')
		parser.add_argument('--T-0', default=self.config['T_0'], type=float, dest='T_0', help='Reference temperature, °C (default: %(default)s)')
		parser.add_argument('--S-0', default=self.config['S_0'], type=float, dest='S_0', help='Reference salinity, PSU (default: %(default)s)')
		parser.add_argument('--ocean-bc', default=self.config['ocean_bc'], dest='ocean_bc', help='Regulates in/out flow at ocean boundary. If a number is given, it will be used as scaling-coefficient of the sinusodial BC on ocean boundary. If a string is given, it will be used as formula for the ocean BC (default: %(default)s)')
		parser.add_argument('--domain', default=self.config['domain'], help='What domain to use, either `square` (1km x 1km) or `custom` (default: %(default)s)')
		parser.add_argument('--domain-size-x', default=self.config['domain_size_x'], type=int, dest='domain_size_x', help='Size of domain in x direction (i.e. width) (default: %(default)s)')
		parser.add_argument('--domain-size-y', default=self.config['domain_size_y'], type=int, dest='domain_size_y', help='Size of domain in y direction (i.e. height) (default: %(default)s)')
		parser.add_argument('--shelf-size-x', default=self.config['shelf_size_x'], type=float, dest='shelf_size_x', help='Size of ice shelf in x direction (i.e. width) (default: %(default)s)')
		parser.add_argument('--shelf-size-y', default=self.config['shelf_size_y'], type=float, dest='shelf_size_y', help='Size of ice shelf in y direction (i.e. height) (default: %(default)s)')
		parser.add_argument('--mesh-resolution', default=self.config['mesh_resolution'], type=int, dest='mesh_resolution', help='Mesh resolution (default: %(default)s) - does not apply to `rectangle` domain')
		parser.add_argument('--mesh-resolution-x', default=self.config['mesh_resolution_x'], type=int, dest='mesh_resolution_x', help='Mesh resolution in x direction (default: %(default)s) - only applies to `rectangle` domain')
		parser.add_argument('--mesh-resolution-y', default=self.config['mesh_resolution_y'], type=int, dest='mesh_resolution_y', help='Mesh resolution in y direction (default: %(default)s) - only applies to `rectangle` domain')
		parser.add_argument('--mesh-resolution-sea-top-y', default=self.config['mesh_resolution_sea_top_y'], type=int, dest='mesh_resolution_sea_y', help='Mesh resolution for sea top beside ice shelf in y direction (default: %(default)s) - only applies to `rectangle` domain')
		parser.add_argument('--store-sol', default=self.config['store_sol'], dest='store_solutions', action='store_true', help='Whether to save iteration solutions for display in Paraview (default: %(default)s)')
		parser.add_argument('--label', default='', help='Label to append to plots folder (default: %(default)s)')
		parser.add_argument('-v', '--verbose', default=self.config['verbose'], dest='verbose', action='store_true', help='Whether to display debug info (default: %(default)s)')
		parser.add_argument('-vv', '--very-verbose', default=self.config['very_verbose'], dest='very_verbose', action='store_true', help='Whether to display debug info from FEniCS as well (default: %(default)s)')
		add_bool_arg(parser, 'plot', default=self.config['plot'], help='Whether to plot solution (default: %(default)s)')

		return parser.parse_args()

	def create_mesh(self):
		""" To create a Mesh, either use one of the built-in meshes https://fenicsproject.org/docs/dolfin/1.4.0/python/demo/documented/built-in_meshes/python/documentation.html or create a custom one.

		Notice that the class PolygonalMeshGenerator has been deprecated so it's no longer available. Instead, we need to call generate_mesh on a custom domain. Custom domains are generated by combining through union, intersection or difference elementary shapes. Custom meshes require the package mshr https://bitbucket.org/benjamik/mshr/src/master/ , which comes if FEniCS is installed system-wide (i.e. through apt, not through Anaconda). Elementary shapes are dolfin.Point, Rectangle, Circle, Ellipse, Polygon: https://bitbucket.org/benjamik/mshr/wiki/browse/API"""

		self.log('Initializing mesh...')

		if self.args.domain == "square":
			self.mesh = UnitSquareMesh(self.args.mesh_resolution, self.args.mesh_resolution)

		if self.args.domain == "custom":
			# general domain geometry: width, height, ice shelf width, ice shelf thickness
			domain_params = [self.args.domain_size_x, self.args.domain_size_y, self.args.shelf_size_x, self.args.shelf_size_y]

			sg = ShelfGeometry(
				domain_params,
				ny_ocean = self.args.mesh_resolution_y,          # layers on "deep ocean" (y-dir)
				ny_shelf = self.args.mesh_resolution_sea_y,      # layers on "ice-shelf thickness" (y-dir)
				nx = self.args.mesh_resolution_x,				 # layers x-dir
			)

			sg.generate_mesh()
			self.mesh = sg.get_fenics_mesh()

			#self.refine_mesh_at_point(Point(self.args.shelf_size_x, self.args.domain_size_y - self.args.shelf_size_y))

			'''fenics_domain = Rectangle(Point(0., 0.), Point(1., 1.)) - \
							Rectangle(Point(0.0, 0.9), Point(0.4, 1.0))
			mesh = generate_mesh(fenics_domain, resolution, "cgal")
			deform_mesh_coords(mesh)

			mesh = refine_mesh_at_point(mesh, Point(0.4, 0.9), domain)'''

		self.log('Initialized mesh: vertexes %d, max diameter %.2f' % (self.mesh.num_vertices(), self.mesh.hmax()), True)

	def mesh_add_sill(self, center, height, length):
		"""Deforms mesh coordinates to create the bottom bump"""

		x = self.mesh.coordinates()[:, 0]
		y = self.mesh.coordinates()[:, 1]

		alpha = 4*height/length**2
		sill_function = lambda x : ((-alpha*(x - center)**2) + height)
		sill_left = center - sqrt(height/alpha)
		sill_right = center + sqrt(height/alpha)

		new_y = [y[i] + sill_function(x[i])*(1-y[i]) if(x[i] < sill_right and x[i] > sill_left) else 0 for i in range(len(y))]
		y = np.maximum(y, new_y)

		self.mesh.coordinates()[:] = np.array([x, y]).transpose()

		#self.sill = {'f':sill_function, 'left':sill_left, 'right':sill_right}
		#self.bd.sill = self.sill

	def refine_mesh_at_point(self, target):
		"""Refines mesh at a given point, taking points in a ball of radius mesh.hmax() around the target.

		A good resource https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1005.html"""

		self.log('Refining mesh at (%.0f, %.0f)' % (target[0], target[1]))

		to_refine = MeshFunction("bool", self.mesh, self.mesh.topology().dim() - 1)
		to_refine.set_all(False)
		mesh = self.mesh #inside `to_refine_subdomain` `self.mesh` does not exist, as `self` is redefined

		class to_refine_subdomain(SubDomain):
			def inside(self, x, on_boundary):
				return ((Point(x) - target).norm() < mesh.hmax())

		D = to_refine_subdomain()
		D.mark(to_refine, True)
		#print(to_refine.array())
		self.mesh = refine(self.mesh, to_refine)

		'''def refine_boundary_mesh(mesh, domain):
		"""Refines mesh on ALL boundary points"""

		boundary_domain = MeshFunction("bool", mesh, mesh.topology().dim() - 1)
		boundary_domain.set_all(False)

		#Get all members of imported boundary and select only the boundary classes (i.e. exclude all other imported functions, such as from fenics).
		#Mark boundary cells as to be refined and do so.
		members = inspect.getmembers(bd, inspect.isclass) #bd is boundary module (included with this code)
		for x in members:
			if 'Bound_' in x[0]:
				obj = getattr(bd, x[0])()
				obj.mark(boundary_domain, True)

		mesh = refine(mesh, boundary_domain)

		log('Refined mesh at boundaries')

		return mesh'''

	def define_function_spaces(self):
		"""Define function spaces and create needed functions for simulation"""

		# Define function spaces
		self.function_spaces.update({
			'V': VectorFunctionSpace(self.mesh, 'CG', 2),
			'Q': FunctionSpace(self.mesh, 'CG', 1),
			'T': FunctionSpace(self.mesh, 'CG', 2),
			'S': FunctionSpace(self.mesh, 'CG', 2)
		})

		# Define functions for solution computation
		self.functions.update({
			'u_n': Function(self.function_spaces.V),
			'u_': Function(self.function_spaces.V),
			'p_n': Function(self.function_spaces.Q),
			'p_': Function(self.function_spaces.Q),
			'T_n': Function(self.function_spaces.T),
			'T_': Function(self.function_spaces.T),
			'S_n': Function(self.function_spaces.S),
			'S_': Function(self.function_spaces.S)
		})

		# Set T and S to reference values to speed up convergence
		self.functions.T_n.assign(interpolate(Constant(self.const.T_0), self.function_spaces.T))
		self.functions.S_n.assign(interpolate(Constant(self.const.S_0), self.function_spaces.S))
		self.functions.p_n.assign(interpolate(Expression('rho_0*g*(1-x[1])', degree=2, rho_0=self.const.rho_0, g=self.const.g), self.function_spaces.Q))

	def define_variational_problems(self):

		# Define trial and test functions
		self.functions.update({
			'u': TrialFunction(self.function_spaces.V),
			'v': TestFunction(self.function_spaces.V),
			'p': TrialFunction(self.function_spaces.Q),
			'q': TestFunction(self.function_spaces.Q),
			'T': TrialFunction(self.function_spaces.T),
			'T_v': TestFunction(self.function_spaces.T),
			'S': TrialFunction(self.function_spaces.S),
			'S_v': TestFunction(self.function_spaces.S)
		})

		# Define expressions used in variational forms
		U = 0.5*(self.functions.u_n + self.functions.u)
		n = FacetNormal(self.mesh)
		f_T = Constant(0)
		f_S = Constant(0)

		buoyancy = Expression((0, '-g*(1 -alpha*(T_ - T_0) + beta*(S_ - S_0))'), alpha=self.const.alpha, beta=self.const.beta, T_0=self.const.T_0, S_0=self.const.S_0, g=self.const.g, T_=self.functions.T_, S_=self.functions.S_, rho_0=self.const.rho_0, degree=2)

		# Define strain-rate tensor
		def epsilon(u):
			return sym(nabla_grad(u))

		# Define stress tensor
		def sigma(u, p):
			return 2*elem_mult(self.const.nu, epsilon(u)) - p*Identity(len(u))

		# Element-wise multiplication (for viscosity)
		'''def el_mult(u, c):
			if ! (isinstance(u, function.argument.Argument) or isinstance(u, function.function.Function)):
				raise ValueError("First argument is of type %s instead of fenics Function/TrialFunction" % type(u))

			assert type(c) == tuple and size(c) == u.geometric_dimension(), "Second argument should be a constant of same size as first argument function"

			temp = TrialFunction(V) if isinstance(u, function.argument.Argument) else Function(V)
			for i in temp.geometric_dimension():
				temp.sub(i).assign(u.sub(i)*c[i])

			return temp

		c=(1,2)
		#print(el_mult(U,c))
		'''

		def get_matrix_diagonal(mat):
			diag = []
			for i in range(mat.ufl_shape[0]):
				diag.append(mat[i][i])

			return as_vector(diag)

		# Define variational problem for step 1
		F1 = dot((self.functions.u - self.functions.u_n)/self.const.dt, self.functions.v)*dx + \
			 dot(dot(self.functions.u_n, nabla_grad(self.functions.u_n)), self.functions.v)*dx \
		   + inner(sigma(U, self.functions.p_n/self.const.rho_0), epsilon(self.functions.v))*dx \
		   + dot(self.functions.p_n*n/self.const.rho_0, self.functions.v)*ds - dot(elem_mult(self.const.nu, nabla_grad(U))*n, self.functions.v)*ds \
		   - dot(buoyancy, self.functions.v)*dx
		self.stiffness_mats.a1, self.load_vectors.L1 = lhs(F1), rhs(F1)

		# Variational problem for pressure p with approximated velocity u
		F = + dot(nabla_grad(self.functions.p - self.functions.p_n), nabla_grad(self.functions.q))/self.const.rho_0*dx \
			+ div(self.functions.u_)*self.functions.q*(1/self.const.dt)*dx
		self.stiffness_mats.a2, self.load_vectors.L2 = lhs(F), rhs(F)

		# Variational problem for corrected velocity u with pressure p
		F = dot(self.functions.u, self.functions.v)*dx \
			- dot(self.functions.u_, self.functions.v)*dx \
			+ dot(nabla_grad(self.functions.p_ - self.functions.p_n), self.functions.v)/self.const.rho_0*self.const.dt*dx # dx must be last multiplicative factor, it's the measure
		self.stiffness_mats.a3, self.load_vectors.L3 = lhs(F), rhs(F)

		# Variational problem for temperature
		F = dot((self.functions.T - self.functions.T_n)/self.const.dt, self.functions.T_v)*dx \
			+ div(self.functions.u_*self.functions.T)*self.functions.T_v*dx \
			+ dot(elem_mult(get_matrix_diagonal(self.const.nu), grad(self.functions.T)), grad(self.functions.T_v))*dx \
			- f_T*self.functions.T_v*dx

		self.stiffness_mats.a4, self.load_vectors.L4 = lhs(F), rhs(F)

		# Variational problem for salinity
		F = dot((self.functions.S - self.functions.S_n)/self.const.dt, self.functions.S_v)*dx \
			+ div(self.functions.u_*self.functions.S)*self.functions.S_v*dx \
			+ dot(elem_mult(get_matrix_diagonal(self.const.nu), grad(self.functions.S)), grad(self.functions.S_v))*dx \
			- f_S*self.functions.S_v*dx
		self.stiffness_mats.a5, self.load_vectors.L5 = lhs(F), rhs(F)

		self.log('Defined variational problems')

	def boundary_conditions(self):
		"""Applies boundary conditions, different depending on domain.

		Draws boundaries from external module."""

		# In/Out velocity flow sinusodial expression
		try:
			coeff = float(self.const.ocean_bc)
			ocean_bc = str(coeff)+"*sin(2*pi*x[1])"
		except ValueError:
			ocean_bc = self.const.ocean_bc

		if(self.args.domain == 'square'):

			# Define boundaries
			top = self.bd.Bound_Top()
			bottom = self.bd.Bound_Bottom()
			left = self.bd.Bound_Left()
			right = self.bd.Bound_Right()

			# Define boundary conditions
			self.bcu.append(DirichletBC(self.function_spaces.V, Constant((0, 0)), top))
			self.bcu.append(DirichletBC(self.function_spaces.V, Constant((0, 0)), bottom))
			self.bcu.append(DirichletBC(self.function_spaces.V, Constant((0, 0)), left))
			self.bcu.append(DirichletBC(self.function_spaces.V, Expression((ocean_bc, 0), degree = 2), right))

			self.bcp.append(DirichletBC(self.function_spaces.Q, Constant(0), top)) #applying BC on right corner yields problems?

			#self.bcT.append(DirichletBC(T_space, Expression("7*x[1]-2", degree=2), right))

			self.bcT.append(DirichletBC(self.function_spaces.T, Constant("3"), right))
			self.bcT.append(DirichletBC(self.function_spaces.T, Constant("-1.9"), left))

			self.bcS.append(DirichletBC(self.function_spaces.S, Expression("35", degree=2), right))
			self.bcS.append(DirichletBC(self.function_spaces.S, Expression("34.5", degree=2), left))

		elif(self.args.domain == 'custom'):

			# Define boundaries
			sea_top = self.bd.Bound_Sea_Top()
			ice_shelf_bottom = self.bd.Bound_Ice_Shelf_Bottom()
			ice_shelf_right = self.bd.Bound_Ice_Shelf_Right()
			bottom = self.bd.Bound_Bottom()
			left = self.bd.Bound_Left()
			right = self.bd.Bound_Right()

			# Define boundary conditions
			self.bcu.append(DirichletBC(self.function_spaces.V, Expression((ocean_bc, 0), degree = 2), right))
			self.bcu.append(DirichletBC(self.function_spaces.V, Constant((0.0, 0.0)), bottom))
			self.bcu.append(DirichletBC(self.function_spaces.V, Constant((0.0, 0.0)), left))
			self.bcu.append(DirichletBC(self.function_spaces.V.sub(1), Constant(0.0), sea_top))

			self.bcp.append(DirichletBC(self.function_spaces.Q, Constant(0), sea_top)) #applying BC on right corner yields problems?

			self.bcT.append(DirichletBC(self.function_spaces.T, Expression("3", degree=2), right))
			self.bcT.append(DirichletBC(self.function_spaces.T, Expression("-1.9", degree=2), left))

			self.bcS.append(DirichletBC(self.function_spaces.S, Expression("35", degree=2), right))
			self.bcS.append(DirichletBC(self.function_spaces.S, Expression("30", degree=2), left))

			# Only set BCs for ice shelf if shelf is actually present
			if self.args.shelf_size_x > 0 and self.args.shelf_size_y > 0:
				self.bcu.append(DirichletBC(self.function_spaces.V, Constant((0.0, 0.0)), ice_shelf_bottom))
				self.bcu.append(DirichletBC(self.function_spaces.V, Constant((0.0, 0.0)), ice_shelf_right))

				self.bcT.append(DirichletBC(self.function_spaces.T, Expression("-1.9", degree=2), ice_shelf_bottom))
				self.bcT.append(DirichletBC(self.function_spaces.T, Expression("-1.9", degree=2), ice_shelf_right))

				self.bcS.append(DirichletBC(self.function_spaces.S, Expression("30", degree=2), ice_shelf_bottom))
				self.bcS.append(DirichletBC(self.function_spaces.S, Expression("30", degree=2), ice_shelf_right))

		'''
		# Enable to check subdomains are properly marked.

		sub_domains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
		left.mark(sub_domains, 1)
		bottom.mark(sub_domains, 2)
		right.mark(sub_domains, 3)
		sea_top.mark(sub_domains, 4)
		ice_shelf_right.mark(sub_domains, 5)
		ice_shelf_bottom.mark(sub_domains, 6)
		File("boundaries.pvd") << sub_domains
		'''

	def run_simulation(self):
		"""Actually run the simulation"""

		self.log('Starting simulation')

		if self.args.store_solutions:
			pathlib.Path(self.plot_path + 'paraview/').mkdir(parents=True, exist_ok=True)
			u_pvd = File(self.plot_path + 'paraview/velocity.pvd')
			p_pvd = File(self.plot_path + 'paraview/pressure.pvd')
			T_pvd = File(self.plot_path + 'paraview/temperature.pvd')
			S_pvd = File(self.plot_path + 'paraview/salinity.pvd')

			u_pvd << self.functions.u_
			p_pvd << self.functions.p_
			T_pvd << self.functions.T_
			S_pvd << self.functions.S_

		# Assemble stiffness matrices (a4, a5 need to be assembled at every time step)
		# Load vectors have coefficients which change upon every iteration
		A1 = assemble(self.stiffness_mats.a1)
		A2 = assemble(self.stiffness_mats.a2)
		A3 = assemble(self.stiffness_mats.a3)

		# Apply boundary conditions
		[bc.apply(A1) for bc in self.bcu]
		[bc.apply(A2) for bc in self.bcp]

		# Time-stepping
		iterations_n = self.args.steps_n*int(self.args.final_time)
		rounded_iterations_n = pow(10, (round(math.log(self.args.steps_n*int(self.args.final_time), 10))))
		start, last_run = 0, 0

		for n in range(iterations_n):

			# Applying IPC splitting scheme (IPCS)
			# Step 1: Tentative velocity step
			b1 = assemble(self.load_vectors.L1)
			[bc.apply(b1) for bc in self.bcu]
			solve(A1, self.functions.u_.vector(), b1)

			# Step 2: Pressure correction step
			b2 = assemble(self.load_vectors.L2)
			[bc.apply(b2) for bc in self.bcp]
			solve(A2, self.functions.p_.vector(), b2)

			# Step 3: Velocity correction step
			b3 = assemble(self.load_vectors.L3)
			solve(A3, self.functions.u_.vector(), b3)

			# Step 4: Temperature step
			# Reassemble stiffness matrix and re-set BC, same for load vector, as coefficients change due to u_
			b4 = assemble(self.load_vectors.L4)
			[bc.apply(b4) for bc in self.bcT]
			A4 = assemble(self.stiffness_mats.a4)
			[bc.apply(A4) for bc in self.bcT]
			solve(A4, self.functions.T_.vector(), b4)

			# Step 5: Salinity step
			# Reassemble stiffness matrix and re-set BC, same for load vector, as coefficients change due to u_
			b5 = assemble(self.load_vectors.L5)
			[bc.apply(b5) for bc in self.bcS]
			A5 = assemble(self.stiffness_mats.a5)
			[bc.apply(A5) for bc in self.bcS]
			solve(A5, self.functions.S_.vector(), b5)

			u_diff = np.linalg.norm(self.functions.u_.vector().get_local() - self.functions.u_n.vector().get_local())
			p_diff = np.linalg.norm(self.functions.p_.vector().get_local() - self.functions.p_n.vector().get_local())
			T_diff = np.linalg.norm(self.functions.T_.vector().get_local() - self.functions.T_n.vector().get_local())
			S_diff = np.linalg.norm(self.functions.S_.vector().get_local() - self.functions.S_n.vector().get_local())

			# Even if verbose, get progressively less verbose with the order of number of iterations
			if(self.args.very_verbose or (rounded_iterations_n < 1000 or (rounded_iterations_n >= 1000 and n % (rounded_iterations_n/100) == 0))):
				last_run = time.time() - start
				#last_run = (last_run*(n/100 - 1) + (time.time() - start))/(n/100) if n != 0 else 0
				eta = round(last_run*(iterations_n - n))/100 if last_run != 0 else '?'

				self.log('Step %d of %d (ETA: ~ %s seconds)' % (n, iterations_n, eta))

				self.log("||u|| = %s, ||u||_8 = %s, ||u-u_n|| = %s, ||p|| = %s, ||p||_8 = %s, ||p-p_n|| = %s, ||T|| = %s, ||T||_8 = %s, ||T-T_n|| = %s, ||S|| = %s, ||S||_8 = %s, ||S - S_n|| = %s" % ( \
					round(norm(self.functions.u_, 'L2'), self.round_precision), round(norm(self.functions.u_.vector(), 'linf'), self.round_precision), round(u_diff, self.round_precision), \
					round(norm(self.functions.p_, 'L2'), self.round_precision), round(norm(self.functions.p_.vector(), 'linf'), self.round_precision), round(p_diff, self.round_precision), \
					round(norm(self.functions.T_, 'L2'), self.round_precision), round(norm(self.functions.T_.vector(), 'linf'), self.round_precision), round(T_diff, self.round_precision), \
					round(norm(self.functions.S_, 'L2'), self.round_precision), round(norm(self.functions.S_.vector(), 'linf'), self.round_precision), round(S_diff, self.round_precision)) \
				)

				start = time.time()

			if self.args.store_solutions:
				u_pvd << self.functions.u_
				p_pvd << self.functions.p_
				T_pvd << self.functions.T_
				S_pvd << self.functions.S_

			convergence_threshold = 10**(self.args.simulation_precision)
			if all(diff < convergence_threshold for diff in [u_diff, p_diff, T_diff, S_diff]):
				self.log('--- Stopping simulation at step %d: all variables reached desired precision ---' % n, True)

				self.log("||u|| = %s, ||u||_8 = %s, ||u-u_n|| = %s, ||p|| = %s, ||p||_8 = %s, ||p-p_n|| = %s, ||T|| = %s, ||T||_8 = %s, ||T-T_n|| = %s, ||S|| = %s, ||S||_8 = %s, ||S - S_n|| = %s" % ( \
					round(norm(self.functions.u_, 'L2'), self.round_precision), round(norm(self.functions.u_.vector(), 'linf'), self.round_precision), round(u_diff, self.round_precision), \
					round(norm(self.functions.p_, 'L2'), self.round_precision), round(norm(self.functions.p_.vector(), 'linf'), self.round_precision), round(p_diff, self.round_precision), \
					round(norm(self.functions.T_, 'L2'), self.round_precision), round(norm(self.functions.T_.vector(), 'linf'), self.round_precision), round(T_diff, self.round_precision), \
					round(norm(self.functions.S_, 'L2'), self.round_precision), round(norm(self.functions.S_.vector(), 'linf'), self.round_precision), round(S_diff, self.round_precision)) \
				)

				break

			if norm(self.functions.u_, 'L2') != norm(self.functions.u_, 'L2'):
				self.log('--- Stopping simulation at step %d: velocity is NaN! ---' % n, True)
				break

			# Set solutions for next time-step
			#y = interpolate(Expression('-g*(1-x[1])', degree=2, g=self.const.g), self.function_spaces.Q)
			self.functions.u_n.assign(self.functions.u_)
			self.functions.p_n.assign(self.functions.p_ )#+ self.const.rho_0*y)
			self.functions.T_n.assign(self.functions.T_)
			self.functions.S_n.assign(self.functions.S_)

		if(self.args.plot == True):
			self.plot_solution()

		os.system('xdg-open "' + self.plot_path + '"')

	def plot_solution(self):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		pl = plot(self.mesh, title = 'Mesh')
		ax.set_aspect('auto')
		plt.savefig(self.plot_path + 'mesh.png', dpi = 800)
		plt.close()

		fig = plt.figure()
		ax = fig.add_subplot(111)
		pl = plot(self.functions.u_, title='Velocity (km/h)')
		plt.colorbar(pl)
		ax.set_aspect('auto')
		plt.savefig(self.plot_path + 'velxy.png', dpi = 800)
		plt.close()

		fig = plt.figure()
		ax = fig.add_subplot(111)
		pl = plot(self.functions.u_[0], title='Velocity X-component (km/h)')
		plt.colorbar(pl)
		ax.set_aspect('auto')
		plt.savefig(self.plot_path + 'velx.png', dpi = 500)
		plt.close()

		fig = plt.figure()
		ax = fig.add_subplot(111)
		pl = plot(self.functions.u_[1], title='Velocity Y-component (km/h)')
		plt.colorbar(pl)
		ax.set_aspect('auto')
		plt.savefig(self.plot_path + 'vely.png', dpi = 500)
		plt.close()

		y = Expression('x[1]', degree = 2)
		p_to_plot = self.functions.p_ #- self.const.rho_0*self.const.g*y #p is redefined in variational problem to include a rho_0*g*y term
		fig = plt.figure()
		ax = fig.add_subplot(111)
		pl = plot(p_to_plot, title='Pressure (Pa)')
		plt.colorbar(pl)
		ax.set_aspect('auto')
		plt.savefig(self.plot_path + 'pressure.png', dpi = 500)
		plt.close()

		fig = plt.figure()
		ax = fig.add_subplot(111)
		pl = plot(self.functions.T_, title='Temperature (°C)')
		plt.colorbar(pl)
		ax.set_aspect('auto')
		plt.savefig(self.plot_path + 'temperature.png', dpi = 500)
		plt.close()

		fig = plt.figure()
		ax = fig.add_subplot(111)
		pl = plot(self.functions.S_, title='Salinity (PSU)')
		plt.colorbar(pl)
		ax.set_aspect('auto')
		plt.savefig(self.plot_path + 'salinity.png', dpi = 500)
		plt.close()

		'''fig = plot(div(u_), title='Velocity divergence')
		plt.colorbar(fig)
		plt.savefig(plot_path + 'div_u.png', dpi = 500)
		plt.close()'''

		'''bmesh = BoundaryMesh(mesh, "exterior", True)
		boundary = bmesh.coordinates()
		BC = { 'x': [], 'y': [], 'ux': [], 'uy': [] }
		fig = plt.figure()
		for i in range(len(boundary)):
			BC['x'].append(boundary[i][0])
			BC['y'].append(boundary[i][1])
			BC['ux'].append(u_(boundary[i])[0])
			BC['uy'].append(u_(boundary[i])[1])
		plt.quiver(BC['x'], BC['y'], BC['ux'], BC['uy'])
		plt.savefig(plot_path + 'boundaryvel.png', dpi = 500)
		plt.close()'''

	'''def plot_boundary_conditions():
		_u_1, _u_2 = u_.split(True)

		bmesh = BoundaryMesh(mesh, "exterior", True)
		boundarycoords = bmesh.coordinates()

		rightboundary=[]
		xvelrightboundary=[]
		BC = np.empty(shape=(len(boundarycoords), 2)) #https://stackoverflow.com/a/569063
		for i in range(len(boundarycoords)):

			if boundarycoords[i][0]==1:
				BC[i][1] = boundarycoords[i][1]
				BC[i][0] = (_u_1(boundarycoords[i]))/10+1
			else:
				BC[i][0] = boundarycoords[i][0]
				BC[i][1] = boundarycoords[i][1]

		fig6 = plt.figure()
		plt.scatter(BC[:,0], BC[:,1])
		plt.savefig('boundary_conditions.png', dpi = 300)
		plt.close()

		#plt.scatter(boundarycoords[:,0], boundarycoords[:,1])
		'''

	def log(self, message, always = False):
		if(self.args.verbose == True or always == True):
			print('* %s' % message, flush=True)
			self.log_file.write(message + '\n')

	def sigint_handler(self, sig, frame):
		if(self.args.plot == True):
			self.log('Simulation stopped -- jumping to plotting before exiting...')
			self.plot_solution()
			os.system('xdg-open "' + self.plot_path + '"')

		sys.exit(0)

	def __del__(self):
		try:
			self.log('--- Finished at %s --- ' % str(datetime.now()), True)
			self.log('--- Duration: %s seconds --- ' % round((time.time() - self.start_time), 2), True)
		except: # avoid errors when called only with -h flag
			pass

class Bunch(object):
	"""Allows to syntaxically handle dictionaries as objects, for pure convenience of writing.
	Instead of the clumsy `adict['pippo']`, use `adict.pippo`."""

	def __init__(self, adict):
		self.__dict__.update(adict)

	def update(self, adict):
		self.__dict__.update(adict)

	def __str__(self):
		return str(self.__dict__)
