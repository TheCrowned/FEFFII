"""
FEniCS tutorial demo program: Heat equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  u'= Laplace(u) + f  in the unit square
  u = u_D             on the boundary
  u = u_0             at t = 0

  u = 1 + x^2 + alpha*y^2 + \beta*t
  f = beta - 2 - 2*alpha
"""

from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

TT = 1.0            # final time
num_steps = 20     # number of time steps
dt = 1 / num_steps # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta
K = 100

# Create mesh and define function space
nx = ny = 50
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
#T_D = Expression('30', degree=2, alpha=alpha, beta=beta, t=0)

def top(x, on_boundary):
	return on_boundary and near(x[1], 1)
def bottom(x, on_boundary):
	if on_boundary and near(x[1], 0):
		return True

ux_sin = "-(30)*sin(2*pi*x[0])"
ux_sin2 = "-(30)*sin(2*pi*x[0])"

bct = []
bct.append(DirichletBC(V, Expression(ux_sin, degree=2), top))
bct.append(DirichletBC(V, Expression(ux_sin2, degree=2), bottom))

# Define initial value
#T_n = interpolate(T_D, V)
#u_n = project(u_D, V)
T_n = Function(V)
T_ = Function(V)

# Define variational problem
T = TrialFunction(V)
v = TestFunction(V)
f = Constant(0) #Constant(beta - 2 - 2*alpha)
u = Constant((0, 0))
k = Constant(dt)

F = T*v*dx + div(u*T)*v*k*dx + k*K*dot(grad(T), grad(v))*dx - (T_n + k*f)*v*dx
a, L = lhs(F), rhs(F)

A = assemble(a)
[bc.apply(A) for bc in bct]

# Time-stepping
t = 0
iterations_n = num_steps*int(TT)
for n in range(iterations_n):

	# Update current time
	t += dt
	#T_D.t = t

	# Compute solution
	b = assemble(L)
	[bc.apply(b) for bc in bct]
	solve(A, T_.vector(), b)

	# Plot solution
	fig = plot(T_)
	print('NOW PLOT %d' % n)
	plt.colorbar(fig)
	plt.savefig('heat%d.png' % n, dpi = 300)
	plt.close()

	# Compute error at vertices
	#T_e = interpolate(T_D, V)
	#error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
	#print('t = %.2f: error = %.3g' % (t, error))

	# Update previous solution
	T_n.assign(T_)
