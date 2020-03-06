#!/usr/bin/env python
# coding: utf-8

"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for channel flow on the unit square using the
Incremental Pressure Correction Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
								 div(u) = 0
"""

from __future__ import print_function
from datetime import datetime
from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import sys, getopt, argparse, os
import math
import inspect
from importlib import import_module
import time

def parse_commandline_args():
	global args
	
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
	parser.add_argument('--final-time', default=10.0, type=float, dest='final_time', help='How long to run the simulation for (default: %(default)s)')
	parser.add_argument('--steps-n', default=10, type=int, dest='steps_n', help='How many steps each of the "seconds" is made of (default: %(default)s)')
	parser.add_argument('--viscosity', default=0.36, type=float, help='Viscosity (default: %(default)s)')
	parser.add_argument('--density', default=1000, type=int, help='Density (default: %(default)s)')
	parser.add_argument('--domain', default='custom', help='What domain to use, either `square` or `custom` (default: %(default)s)')
	parser.add_argument('--mesh-resolution', default=16, type=int, dest='mesh_resolution', help='Mesh resolution (default: %(default)s)')
	parser.add_argument('-v', '--verbose', default=False, dest='verbose', action='store_true', help='Whether to display debug info (default: %(default)s)')
	parser.add_argument('-vv', '--very-verbose', default=False, dest='very_verbose', action='store_true', help='Whether to display debug info from FEniCS as well (default: %(default)s)')
	add_bool_arg(parser, 'plot', default=True, help='Whether to plot solution (default: %(default)s)')
	add_bool_arg(parser, 'plot-BC', default=False, dest='plot_BC', help='Wheher to plot boundary conditions (default: %(default)s)')
	args = parser.parse_args()

def deform_mesh_coords(mesh):
	x = mesh.coordinates()[:, 0]
	y = mesh.coordinates()[:, 1]
	
	new_y = [y[i] + 0.1*np.sin(3*pi*(x[i]-(3/8)))*(1-y[i]) if(x[i] < 0.75 and x[i] > 0.3) else 0 for i in range(len(y)) ]
	y = np.maximum(y, new_y)

	mesh.coordinates()[:] = np.array([x, y]).transpose()
	
	
#good resource https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1005.html
def refine_mesh_at_point(mesh, target, domain):
	log('Refining mesh at (%.0f, %.0f)' % (target[0], target[1]))
	
	to_refine = MeshFunction("bool", mesh, mesh.topology().dim() - 1)
	to_refine.set_all(False)
	
	class to_refine_subdomain(SubDomain):
		def inside(self, x, on_boundary):
			return ((Point(x) - target).norm() < mesh.hmax())

	D = to_refine_subdomain()
	D.mark(to_refine, True)
	#print(to_refine.array())
	mesh = refine(mesh, to_refine)		
	
	return mesh
	
'''
Refines mesh on ALL boundary points
'''
def refine_boundary_mesh(mesh, domain):
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
	
	return mesh
	
def initialize_mesh( domain, resolution ):
	""" To create a Mesh, either use one of the built-in meshes https://fenicsproject.org/docs/dolfin/1.4.0/python/demo/documented/built-in_meshes/python/documentation.html or create a custom one. 
	
	Notice that the class PolygonalMeshGenerator has been deprecated so it's no longer available. Instead, we need to call generate_mesh on a custom domain. Custom domains are generated by combining through union, intersection or difference elementary shapes. Custom meshes require the package mshr https://bitbucket.org/benjamik/mshr/src/master/ , which comes if FEniCS is installed system-wide (i.e. through apt, not through Anaconda). Elementary shapes are, which are dolfin.Point, Rectangle, Circle, Ellipse, Polygon: https://bitbucket.org/benjamik/mshr/wiki/browse/API
	
	The two next cells are **alternatives** implementing a built-in mesh over a unit square or custom one."""
	
	global mesh
	
	if domain == "square":
		mesh = UnitSquareMesh(resolution, resolution)
		
	elif domain == 'custom':
		fenics_domain = Rectangle(Point(0., 0.), Point(1., 1.)) - \
						Rectangle(Point(0.0, 0.9), Point(0.4, 1.0)) 
		mesh = generate_mesh(fenics_domain, resolution, "cgal")
		deform_mesh_coords(mesh)

	log('Initialized mesh')
	
	#mesh = refine_boundary_mesh(mesh, domain)
	
	#refining the mesh at sharp vertexes seems enough, no need to refine the whole boundaries.
	#For density=1000, domain='custom', final_time=10.0, mesh_resolution=16, plot=True, plot_BC=False, steps_n=10, verbose=True, very_verbose=False, viscosity=100, **{'plot-BC': False}
	#it is 7.17 seconds vs 7.84
	
	mesh = refine_mesh_at_point(mesh, Point(0.4, 0.9), domain)
	mesh = refine_mesh_at_point(mesh, Point(0.4, 1), domain)
	mesh = refine_mesh_at_point(mesh, Point(0, 0), domain)
	mesh = refine_mesh_at_point(mesh, Point(1, 0), domain)
	mesh = refine_mesh_at_point(mesh, Point(1, 1), domain)
	mesh = refine_mesh_at_point(mesh, Point(0, 0.9), domain)
	
	print('Final mesh: vertexes %d, max diameter %.2f' % (mesh.num_vertices(), mesh.hmax()))
	
def define_function_spaces():
	global V, Q, T_space, S_space, u_n, u_, p_n, p_, T_n, T_, S_n, S_
	
	# Define function spaces
	V = VectorFunctionSpace(mesh, 'P', 2)
	Q = FunctionSpace(mesh, 'P', 1)
	T_space = FunctionSpace(mesh, 'P', 2)
	S_space = FunctionSpace(mesh, 'P', 2)

	# Define functions for solution computation
	u_n = Function(V)
	u_  = Function(V)
	p_n = Function(Q)
	p_  = Function(Q)
	T_n = Function(T_space)
	T_  = Function(T_space)
	S_n = Function(S_space)
	S_  = Function(S_space)

def boundary_conditions():
	global bcu, bcp, bcT, bcS
	
	bcu = []
	bcp = []
	bcT = []
	bcS = []
	
	# In/Out flow sinusodial expression
	ux_sin = "-(0.3)*sin(2*pi*x[1])"
	
	if(args.domain == 'square'):
		
		# Define boundaries
		top = bd.Bound_Top()
		bottom = bd.Bound_Bottom()
		left = bd.Bound_Left()
		right = bd.Bound_Right()
	
		# Define boundary conditions
		bcu.append(DirichletBC(V, Expression((ux_sin, 0), degree = 2), right))
		bcu.append(DirichletBC(V, Constant((0.0, 0.0)), bottom))
		bcu.append(DirichletBC(V, Constant((0.0, 0.0)), left))
		bcu.append(DirichletBC(V.sub(1), Constant(0.0), top))
		
		bcp.append(DirichletBC(Q, Constant(0), top))
		
		bcT.append(DirichletBC(T_space, Expression("7*x[1]-2", degree=2), right))
		
		bcS.append(DirichletBC(S_space, Expression("5", degree=2), right))
		bcS.append(DirichletBC(S_space, Expression("0", degree=2), left))
	
	elif(args.domain == 'custom'):
		
		# Define boundaries
		sea_top = bd.Bound_Sea_Top()
		ice_shelf_bottom = bd.Bound_Ice_Shelf_Bottom()
		ice_shelf_right = bd.Bound_Ice_Shelf_Right()
		bottom = bd.Bound_Bottom()
		left = bd.Bound_Left()
		right = bd.Bound_Right()
		
		# Define boundary conditions
		bcu.append(DirichletBC(V, Expression((ux_sin, 0), degree = 2), right))
		bcu.append(DirichletBC(V, Constant((0.0, 0.0)), bottom))
		bcu.append(DirichletBC(V, Constant((0.0, 0.0)), left))
		bcu.append(DirichletBC(V, Constant((0.0, 0.0)), ice_shelf_bottom))
		bcu.append(DirichletBC(V, Constant((0.0, 0.0)), ice_shelf_right))
		bcu.append(DirichletBC(V.sub(1), (0.0), sea_top))
		
		bcp.append(DirichletBC(Q, Constant(0), sea_top))
		
		bcT.append(DirichletBC(T_space, Expression("3", degree=2), right))
		bcT.append(DirichletBC(T_space, Expression("-1.9", degree=2), left))
		bcT.append(DirichletBC(T_space, Expression("-1.9", degree=2), ice_shelf_bottom))
		bcT.append(DirichletBC(T_space, Expression("-1.9", degree=2), ice_shelf_right))
		
		bcS.append(DirichletBC(S_space, Expression("35", degree=2), right))
		bcS.append(DirichletBC(S_space, Expression("0", degree=2), left))
		bcS.append(DirichletBC(S_space, Expression("0", degree=2), ice_shelf_bottom))
		bcS.append(DirichletBC(S_space, Expression("0", degree=2), ice_shelf_right))

def define_variational_problems():
	global a1, a2, a3, a4, a5, L1, L2, L3, L4, L5
	
	# Define trial and test functions
	u = TrialFunction(V)
	v = TestFunction(V)
	p = TrialFunction(Q)
	q = TestFunction(Q)
	T = TrialFunction(T_space)
	T_v = TestFunction(T_space)
	S = TrialFunction(S_space)
	S_v = TestFunction(S_space)
	
	# Define expressions used in variational forms
	U   = 0.5*(u_n + u)
	n   = FacetNormal(mesh)
	g   = 1.27*10**5
	f_T = Constant(0)
	f_S = Constant(0)
	dt  = Constant(dt_scalar)
	nu  = Constant(3.6*10**-1)
	rho_0 = Constant(1.028*10**12) #https://en.wikipedia.org/wiki/Seawater#/media/File:WaterDensitySalinity.png
	alpha = Constant(10**(-4)) 
	beta = Constant(7.6*10**(-4)) 
	T_0 = Constant(1)
	S_0 = Constant(35)
	buoyancy = Expression((0, 'g*(-alpha*(T_ - T_0) + beta*(S_ - S_0))'), alpha=alpha, beta=beta, T_0=T_0, S_0=S_0, g=Constant(g), T_=T_, S_=S_, degree=2)
	
	# Variational problem for velocity u with pression p_n from previous step
	F = dot((u - u_n)/dt, v)*dx \
		+ dot(dot(u_n, nabla_grad(u_n)), v)*dx \
		+ inner(2*nu*sym(nabla_grad(U)), sym(nabla_grad(v)))*dx \
		- inner((p_n/rho_0)*Identity(len(U)), sym(nabla_grad(v)))*dx \
		+ dot((p_n/rho_0)*n, v)*ds \
		- dot(nu*nabla_grad(U)*n, v)*ds \
		- dot(buoyancy, v)*dx 
	a1, L1 = lhs(F), rhs(F)

	# Variational problem for pressure p with approximated velocity u
	a2 = dot(nabla_grad(p), nabla_grad(q))*dx
	L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - dot(div(u_), q)/dt*dx

	# Variational problem for corrected velocity u with pressure p
	a3 = dot(u, v)*dx
	L3 = dot(u_, v)*dx - dot(nabla_grad(p_ - p_n), v)*dt*dx # dx must be last multiplicative factor, it's the measure
	
	# Variational problem for temperature
	F = dot((T - T_n)/dt, T_v)*dx \
		+ div(u_*T)*T_v*dx \
		+ nu*dot(grad(T), grad(T_v))*dx \
		- f_T*T_v*dx
	a4, L4 = lhs(F), rhs(F)
	
	# Variational problem for salinity
	F = dot((S - S_n)/dt, S_v)*dx \
		+ div(u_*S)*S_v*dx \
		+ nu*dot(grad(S), grad(S_v))*dx \
		- f_S*S_v*dx
	a5, L5 = lhs(F), rhs(F)
	
	log('Defined variational problems')
	
def run_simulation():
	global u_, p_, T_, S_
	
	log('Starting simulation')
	
	# Time-stepping
	iterations_n = num_steps*int(final_time)
	rounded_iterations_n = pow(10, (round(math.log(num_steps*int(final_time), 10))))
	
	#file = File('temp.pvd')
	#vel = File('vel.pvd')
	
	# Assemble stiffness matrices (a4, a5 need to be assembled at every time step) and load vectors (except those whose coefficients change every iteration)
	A1 = assemble(a1)
	A2 = assemble(a2)
	A3 = assemble(a3)
	b4 = assemble(L4)
	b5 = assemble(L5)
	
	# Apply boundary conditions
	[bc.apply(A1) for bc in bcu]
	[bc.apply(A3) for bc in bcu]
	[bc.apply(A2) for bc in bcp]
	[bc.apply(b4) for bc in bcT]
	[bc.apply(b5) for bc in bcS]
	
	for n in range(iterations_n):
		
		# Even if verbose, get progressively less verbose with the order of number of iterations
		if(rounded_iterations_n < 1000 or (rounded_iterations_n >= 1000 and n % (rounded_iterations_n/100) == 0)):
			log('Step %s of %s' % (n, iterations_n))

		# Step 1: Tentative velocity step
		b1 = assemble(L1)
		[bc.apply(b1) for bc in bcu]
		solve(A1, u_.vector(), b1)
		
		# Step 2: Pressure correction step
		b2 = assemble(L2)
		[bc.apply(b2) for bc in bcp]
		solve(A2, p_.vector(), b2)

		# Step 3: Velocity correction step
		b3 = assemble(L3)
		[bc.apply(b3) for bc in bcu]
		solve(A3, u_.vector(), b3)
		
		# Step 4: Temperature step
		A4 = assemble(a4) # Reassemble stiffness matrix and re-set BC, as coefficients change due to u_
		[bc.apply(A4) for bc in bcT]
		solve(A4, T_.vector(), b4)
		
		# Step 5: Salinity step
		A5 = assemble(a5) # Reassemble stiffness matrix and re-set BC, as coefficients change due to u_
		[bc.apply(A5) for bc in bcS]
		solve(A5, S_.vector(), b5)
		
		print("||u|| = %s, ||u-u_n|| = %s, ||p|| = %s, ||p-p_n|| = %s, ||T|| = %s, ||T-T_n|| = %s, ||S|| = %s, ||S - S_n|| = %s" % ( \
			round(np.linalg.norm(u_.vector().get_local()), 2), \
			round(np.linalg.norm(u_.vector().get_local() - u_n.vector().get_local()), 2), \
			round(np.linalg.norm(p_.vector().get_local()), 2), \
			round(np.linalg.norm(p_.vector().get_local() - p_n.vector().get_local()), 2), \
			round(np.linalg.norm(T_.vector().get_local()), 2), \
			round(np.linalg.norm(T_.vector().get_local() - T_n.vector().get_local()), 2), \
			round(np.linalg.norm(S_.vector().get_local()), 2), \
			round(np.linalg.norm(S_.vector().get_local() - S_n.vector().get_local()), 2)) \
		)

		#file << T_
		#vel << u_

		# Set solutions for next time-step
		u_n.assign(u_)
		p_n.assign(p_)
		T_n.assign(T_)
		S_n.assign(S_)
		
def plot_solution():
	# Plot solution
	#fig = plt.figure(figsize=(80, 60))

	fig1 = plot(u_, title='velocity X,Y')
	plt.colorbar(fig1)
	plt.savefig(plot_path + 'velxy.png', dpi = 500)
	plt.close()

	fig2 = plot(u_[0], title='velocity X')
	plt.colorbar(fig2)
	plt.savefig(plot_path + 'velx.png', dpi = 300)
	plt.close()

	fig3 = plot(u_[1], title='velocity Y')
	plt.colorbar(fig3)
	plt.savefig(plot_path + 'vely.png', dpi = 300)
	plt.close()

	fig4 = plot(p_, title='pressure')
	plt.colorbar(fig4)
	plt.savefig(plot_path + 'pressure.png', dpi = 300)
	plt.close()
	
	fig5 = plot(mesh)
	plt.savefig(plot_path + 'mesh.png', dpi = 800)
	plt.close()
	
	fig6 = plot(T_, title='temperature')
	plt.colorbar(fig6)
	plt.savefig(plot_path + 'temperature.png', dpi = 300)
	plt.close()
	
	fig7 = plot(S_, title='salinity')
	plt.colorbar(fig7)
	plt.savefig(plot_path + 'salinity.png', dpi = 300)
	plt.close()
	
	ub = Function(V)
	bmesh = BoundaryMesh(mesh, "exterior", True)
	boundary = bmesh.coordinates()
	BC = { 'x': [], 'y': [], 'ux': [], 'uy': [] }
	fig = plt.figure()
	for i in range(len(boundary)):
		BC['x'].append(boundary[i][0])
		BC['y'].append(boundary[i][1])
		BC['ux'].append(u_(boundary[i])[0])
		BC['uy'].append(u_(boundary[i])[1])
	plt.quiver(BC['x'], BC['y'], BC['ux'], BC['uy'])
	plt.savefig('boundaryvel.png', dpi = 500)
	plt.close()
	
def plot_boundary_conditions():
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
	
def log(message):
	if(args.verbose == True): print('* %s' % message, flush=True)

if __name__ == '__main__':
	
	start_time = time.time()
	print('--- Started at %s --- ' % str(datetime.now()))
	
	parse_commandline_args()
	
	print('--- Parameters are: ---')
	print(args)
	
	final_time = float(args.final_time)     # final time
	num_steps = int(args.steps_n)   		# number of time steps per time unit
	dt_scalar = 1 / num_steps 						# time step size
	#mu_scalar = float(args.viscosity)      		# kinematic viscosity
	#rho_scalar = float(args.density)       		# density
	
	plot_path = 'plots/plot ' + '--final-time %.0f --steps-n %d --mesh-resolution %d/' % (final_time, num_steps, args.mesh_resolution)

	if(not os.path.isdir(plot_path) and (args.plot or args.plot_BC)):
		os.mkdir(plot_path)

	# tolerance for near() function based on mesh resolution. Otherwise BC are not properly set
	tolerance = pow(10, - round(math.log(args.mesh_resolution, 10)))
	
	#Import correct set of boundaries depending on domain and set tolerance
	bd = import_module('boundaries_' + args.domain)
	bd.tolerance = tolerance
	
	if(args.very_verbose == True):
		set_log_active(True)
		set_log_level(1)
	
	initialize_mesh(args.domain, args.mesh_resolution)
	define_function_spaces()
	start_time = time.time()
	define_variational_problems()
	boundary_conditions()
	run_simulation()
	
	print('--- Finished at %s --- ' % str(datetime.now()))
	print('--- Duration: %s seconds --- ' % round((time.time() - start_time), 2))
	
	if(args.plot == True):
		plot_solution()
	
	if(args.plot_BC == True):
		plot_boundary_conditions()
