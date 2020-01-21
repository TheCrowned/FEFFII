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
import sys, getopt, argparse
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
	parser.add_argument('--viscosity', default=100, type=int, help='Viscosity (default: %(default)s)')
	parser.add_argument('--density', default=1000, type=int, help='Density (default: %(default)s)')
	parser.add_argument('--domain', default='custom', help='What domain to use, either `square` or `custom` (default: %(default)s)')
	parser.add_argument('--mesh-resolution', default=32, type=int, dest='mesh_resolution', help='Mesh resolution (default: %(default)s)')
	parser.add_argument('-v', '--verbose', default=False, dest='verbose', action='store_true', help='Whether to dsplay debug info (default: %(default)s)')
	parser.add_argument('-vv', '--very-verbose', default=False, dest='very_verbose', action='store_true', help='Whether to dsplay debug info from FEniCS as well (default: %(default)s)')
	add_bool_arg(parser, 'plot', default=True, help='Whether to plot solution (default: %(default)s)')
	add_bool_arg(parser, 'plot-BC', default=False, dest='plot_BC', help='Wheher to plot boundary conditions (default: %(default)s)')
	args = parser.parse_args()

def define_variational_problems():
	global A1, A2, A3, L1, L2, L3
	global rho, mu
	
	# Define trial and test functions
	u = TrialFunction(V)
	v = TestFunction(V)
	p = TrialFunction(Q)
	q = TestFunction(Q)
	
	# Define expressions used in variational forms
	U   = 0.5*(u_n + u)
	n   = FacetNormal(mesh)
	g   = 9.81
	f   = Constant((0, -g))
	k   = Constant(dt)
	mu  = Constant(mu)
	rho = Constant(rho)
	
	# Define strain-rate tensor
	def epsilon(u):
		return sym(nabla_grad(u))

	# Define stress tensor
	def sigma(u, p):
		return 2*mu*epsilon(u) - p*Identity(len(u))

	# Define variational problem for step 1
	F1 = rho*dot((u - u_n) / k, v)*dx +      rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx      + inner(sigma(U, p_n), epsilon(v))*dx      + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds      - dot(f, v)*dx
	a1 = lhs(F1)
	L1 = rhs(F1)

	# Define variational problem for step 2
	a2 = dot(nabla_grad(p), nabla_grad(q))*dx
	L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

	# Define variational problem for step 3
	a3 = dot(u, v)*dx
	L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx
	
	# Assemble matrices
	A1 = assemble(a1)
	A2 = assemble(a2)
	A3 = assemble(a3)
	
	log('Defined variational problems')

def deform_mesh_coords(mesh):
	x = mesh.coordinates()[:, 0]
	y = mesh.coordinates()[:, 1]
	
	new_y = [y[i] + 0.1*np.sin(3*pi*(x[i]-(3/8)))*(1-y[i]) if(x[i] < 0.75 and x[i] > 0.3) else 0 for i in range(len(y)) ]
	y = np.maximum(y, new_y)

	mesh.coordinates()[:] = np.array([x, y]).transpose()
	
def refine_mesh_at_point(mesh, target, domain):
	to_refine = MeshFunction("bool", mesh, mesh.topology().dim() - 1)
	to_refine.set_all(False)
	
	to_refine_vertexes = []
	for v in vertices(mesh):
		if((v.point()-target).norm() < (1/2)*mesh.hmax()):
			to_refine_vertexes.append(v.point().array().tolist())
			#print(type(v.point().array().tolist()))

	class dummy_subdomain(SubDomain):
		def inside(self, x, on_boundary):
			#print (Point(x))
			#print (Point(x).array())
			#print (type(Point(x).array()))
			#print (to_refine_vertexes)
			#print (type(to_refine_vertexes))
			#print (type(to_refine_vertexes[0]))
			if (Point(x).array().tolist() in to_refine_vertexes):
				print(x)
				#print((Point(x)-target).norm())
				return True

	print (to_refine_vertexes)
	D = dummy_subdomain()
	D.mark(to_refine, True)
	
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
	fig5 = plot(mesh)
	plt.savefig('mesh1.png', dpi = 800)
	plt.close()
	#refining the mesh at sharp vertexes seems enough. With refinement over whole boundaries result is better though, but slower. For
	#density=1000, domain='custom', final_time=10.0, mesh_resolution=30, plot=True, plot_BC=False, steps_n=10, verbose=True, very_verbose=False, viscosity=100, **{'plot-BC': False}
	#it is 27.12 seconds vs 31.18
	#mesh = refine_boundary_mesh(mesh, domain) #looks like it is needed as well, or solution goes astray
	print(mesh.hmax())
	mesh = refine_mesh_at_point(mesh, Point(0.4, 0.9), domain)
	#mesh = refine_mesh_at_point(mesh, Point(0, 0), domain)
	#mesh = refine_mesh_at_point(mesh, Point(0, 1), domain)
	#mesh = refine_mesh_at_point(mesh, Point(1, 0), domain)
	#mesh = refine_mesh_at_point(mesh, Point(1, 1), domain)
	#mesh = refine_mesh_at_point(mesh, Point(0, 0.9), domain)
	print(mesh.num_vertices())
	
def define_function_spaces():
	global V, Q, u_n, u_, p_n, p_
	
	# Define function spaces
	V = VectorFunctionSpace(mesh, 'P', 2)
	Q = FunctionSpace(mesh, 'P', 1)

	# Define functions for solution computation
	u_n = Function(V)
	u_  = Function(V)
	p_n = Function(Q)
	p_  = Function(Q)

def boundary_conditions():
	global bcu, bcp
	
	bcu = []
	bcp = []
	
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
		bcu.append(DirichletBC(V.sub(0), (0.0), sea_top))
		
		bcp.append(DirichletBC(Q, Constant(0), sea_top))
	
	# Apply boundary conditions to matrices
	[bc.apply(A1) for bc in bcu]
	[bc.apply(A2) for bc in bcp]
	
def run_simulation():
	global u_, p_
	
	log('Starting simulation')
	
	# Time-stepping
	iterations_n = num_steps*int(T)
	rounded_iterations_n = pow(10, (round(math.log(num_steps*int(T), 10))))
	t = 0
	for n in range(iterations_n):
		
		# Even if verbose, get progressively less verbose with the order of number of iterations
		if(rounded_iterations_n < 1000 or (rounded_iterations_n >= 1000 and n % (rounded_iterations_n/100) == 0)):
			log('Step %s of %s' % (n, iterations_n))
			
		t += dt

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
		solve(A3, u_.vector(), b3)

		# Compute error
		#u_e = Expression(('4*x[1]*(1.0 - x[1])', '0'), degree=2)
		#u_e = interpolate(u_e, V)
		#error = np.abs(u_e.vector().get_local() - u_.vector().get_local()).max()
		#print('t = %.2f: error = %.3g' % (t, error))
		#print('max u:', u_.vector().get_local().max())

		# Update previous solution
		u_n.assign(u_)
		p_n.assign(p_)
	
	return u_, p_
		
def plot_solution():
	# Plot solution
	#fig = plt.figure(figsize=(80, 60))
	
	fig5 = plot(mesh)
	plt.savefig('mesh.png', dpi = 800)
	plt.close()

	fig1 = plot(u_, title='velocity X,Y')
	plt.savefig('velxy.png', dpi = 300)
	plt.close()

	fig2 = plot(u_[0], title='velocity X')
	plt.colorbar(fig2)
	plt.savefig('velx.png', dpi = 300)
	plt.close()

	fig3 = plot(u_[1], title='velocity Y')
	plt.colorbar(fig3)
	plt.savefig('vely.png', dpi = 300)
	plt.close()

	fig4 = plot(p_, title='pressure')
	plt.colorbar(fig4)
	plt.savefig('pressure.png', dpi = 300)
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
	
	T = float(args.final_time)      # final time
	num_steps = int(args.steps_n)   # number of time steps per time unit
	dt = 1 / num_steps 				# time step size
	mu = float(args.viscosity)      # kinematic viscosity
	rho = float(args.density)       # density
	
	# tolerance for near() function based on mesh resolution
	tolerance = pow(10, - round(math.log(args.mesh_resolution, 10)))
	
	#Import correct set of boundaries depending on domain and set tolerance
	bd = import_module('boundaries_' + args.domain)
	bd.tolerance = tolerance
	
	if(args.very_verbose == True):
		set_log_active(True)
		set_log_level(1)
	
	initialize_mesh( args.domain, args.mesh_resolution )
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
