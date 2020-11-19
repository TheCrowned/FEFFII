from fenics import *
import numpy as np
import matplotlib.pyplot as plt

T = 1.0            # final time
num_steps = 50000   # number of time steps
dt = T / num_steps # time step size

# Create mesh
mesh = UnitSquareMesh(30,30)
#plot(mesh)
#plt.show()

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)
T_space = FunctionSpace(mesh, 'P', 2)
S_space = FunctionSpace(mesh, 'P', 2)

# Define boundaries
left   = 'near(x[0], 0)'
right  = 'near(x[0], 1)'
bottom = 'near(x[1], 0)'
top    = 'near(x[1], 1)'

# Define boundary conditions
bcu, bcp, bcT, bcS = [], [], [], []
ux_sin = "-(0.3)*sin(2*pi*x[1])"
bcu.append(DirichletBC(V, Constant((0, 0)), bottom))
bcu.append(DirichletBC(V, Constant((1, 0)), top))
bcu.append(DirichletBC(V, Constant((0, 0)), left))
bcu.append(DirichletBC(V, Constant((0, 0)), right))
#bcu.append(DirichletBC(V.sub(0), Constant(0.04), inflow))
#bcu.append(DirichletBC(V.sub(0), Constant(0.04), outflow))
#bcp.append(DirichletBC(Q, Constant(8), inflow))
bcp.append(DirichletBC(Q, Constant(0), 'near(x[1], 0) && near(x[0], 0)', method='pointwise'))

bcT.append(DirichletBC(T_space, Expression("3", degree=2), right))
bcT.append(DirichletBC(T_space, Expression("-1.8", degree=2), left))

bcS.append(DirichletBC(S_space, Expression("35", degree=2), right))
bcS.append(DirichletBC(S_space, Expression("0", degree=2), left))

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

T = TrialFunction(T_space)
T_v = TestFunction(T_space)
S = TrialFunction(S_space)
S_v = TestFunction(S_space)

# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)

T_n = Function(T_space)
T_  = Function(T_space)
S_n = Function(S_space)
S_  = Function(S_space)

# Define expressions used in variational forms
U  = 0.5*(u_n + u)
n  = FacetNormal(mesh)
g  = -9.81
f  = Constant((0, g))
dt  = Constant(dt)
print(dt.values())
g   = 0
nu  = Constant(0.0001)#3.6*10**-1)
rho_0 = Constant(1)

alpha = Constant(10**(-4))
beta = Constant(7.6*10**(-4))
T_0 = Constant(1)
S_0 = Constant(35)
buoyancy = Expression((0, 'g*(-alpha*(T_ - T_0) + beta*(S_ - S_0))'), alpha=alpha, beta=beta, T_0=T_0, S_0=S_0, g=Constant(g), T_=T_, S_=S_, degree=2)

#p_n = Expression('rho*g*(1-x[1])', g=g, rho=rho, degree=2, )

# Define symmetric gradient
def epsilon(u):
	return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
	return 2*nu*epsilon(u) - (1)*p*Identity(len(u))

# Define variational problem for step 1
F1 = dot((u - u_n)/dt, v)*dx + \
	 dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n/rho_0), epsilon(v))*dx \
   + dot(p_n*n/rho_0, v)*ds - dot(nu*nabla_grad(U)*n, v)*ds #\
   #- dot(buoyancy, v)*dx
a1, L1 = lhs(F1), rhs(F1)

# Variational problem for pressure p with approximated velocity u
F = + dot(nabla_grad(p - p_n), nabla_grad(q))/rho_0*dx \
	+ div(u_)*q*(1/dt)*dx
a2, L2 = lhs(F), rhs(F)

# Variational problem for corrected velocity u with pressure p
F = dot(u, v)*dx \
	- dot(u_, v)*dx \
	+ dot(nabla_grad(p_ - p_n), v)/rho_0*dt*dx # dx must be last multiplicative factor, it's the measure
a3, L3 = lhs(F), rhs(F)

'''
# Define variational problem for step 1
F1 = dot((u - u_n) / k, v)*dx \
   + dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds #\
   #- dot(buoyancy, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*(1)*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*(1)*dx - (1/k)*div(u_)*q*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*(1)*dx
'''
# Variational problem for temperature
F = dot((T - T_n)/dt, T_v)*dx \
	+ div(u_*T)*T_v*dx \
	+ nu*dot(grad(T), grad(T_v))*dx
a4, L4 = lhs(F), rhs(F)

# Variational problem for salinity
F = dot((S - S_n)/dt, S_v)*dx \
	+ div(u_*S)*S_v*dx \
	+ nu*dot(grad(S), grad(S_v))*dx
a5, L5 = lhs(F), rhs(F)

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

b4 = assemble(L4)
b5 = assemble(L5)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]
[bc.apply(b4) for bc in bcT]
[bc.apply(b5) for bc in bcS]

# Create XDMF files for visualization output
#xdmffile_u = XDMFFile('navier_stokes_cylinder/velocity.xdmf')
#xdmffile_p = XDMFFile('navier_stokes_cylinder/pressure.xdmf')

# Create time series (for use in reaction_system.py)
#timeseries_u = TimeSeries('navier_stokes_cylinder/velocity_series')
#timeseries_p = TimeSeries('navier_stokes_cylinder/pressure_series')

# Save mesh to file (for use in reaction_system.py)
#File('navier_stokes_cylinder/cylinder.xml.gz') << mesh

# Create progress bar
#progress = Progress('Time-stepping')
#set_log_level(PROGRESS)

# Time-stepping
t = 0
for n in range(num_steps):
	print("%d of %d" % (n, num_steps))

	# Update current time
	t += dt

	# Step 1: Tentative velocity step
	b1 = assemble(L1)
	[bc.apply(b1) for bc in bcu]
	solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

	# Step 2: Pressure correction step
	b2 = assemble(L2)
	[bc.apply(b2) for bc in bcp]
	solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

	# Step 3: Velocity correction step
	b3 = assemble(L3)
	solve(A3, u_.vector(), b3, 'cg', 'sor')

	# Step 4: Temperature step
	#A4 = assemble(a4) # Reassemble stiffness matrix and re-set BC, as coefficients change due to u_
	#[bc.apply(A4) for bc in bcT]
	#solve(A4, T_.vector(), b4)

	# Step 5: Salinity step
	#A5 = assemble(a5) # Reassemble stiffness matrix and re-set BC, as coefficients change due to u_
	#[bc.apply(A5) for bc in bcS]
	#solve(A5, S_.vector(), b5)

	# Plot solution


	# Save solution to file (XDMF/HDF5)
	#xdmffile_u.write(u_, t)
	#xdmffile_p.write(p_, t)

	# Save nodal values to file
	#timeseries_u.store(u_.vector(), t)
	#timeseries_p.store(p_.vector(), t)

	u_diff = np.linalg.norm(u_.vector().get_local() - u_n.vector().get_local())
	p_diff = np.linalg.norm(p_.vector().get_local() - p_n.vector().get_local())
	T_diff = np.linalg.norm(T_.vector().get_local() - T_n.vector().get_local())
	S_diff = np.linalg.norm(S_.vector().get_local() - S_n.vector().get_local())

	print("||u|| = %s, ||u||_8 = %s, ||u-u_n|| = %s, ||p|| = %s, ||p||_8 = %s, ||p-p_n|| = %s, ||T|| = %s, ||T||_8 = %s, ||T-T_n|| = %s, ||S|| = %s, ||S||_8 = %s, ||S - S_n|| = %s" % ( \
		round(norm(u_, 'L2'), 2), round(norm(u_.vector(), 'linf'), 3), round(u_diff, 3), \
		round(norm(p_, 'L2'), 2), round(norm(p_.vector(), 'linf'), 3), round(p_diff, 3), \
		round(norm(T_, 'L2'), 2), round(norm(T_.vector(), 'linf'), 3), round(T_diff, 3), \
		round(norm(S_, 'L2'), 2), round(norm(S_.vector(), 'linf'), 3), round(S_diff, 3)) \
	)

	convergence_threshold = 10**(-5)
	if all(diff < convergence_threshold for diff in [u_diff, p_diff, T_diff, S_diff]):
		print('--- Stopping simulation at step %d: all variables reached desired precision ---' % n, True)
		break

	 # Update previous solution
	u_n.assign(u_)
	p_n.assign(p_)
	T_n.assign(T_)
	S_n.assign(S_)

fig2 = plot(u_, title='velocity X,Y')
plt.colorbar(fig2)
plt.savefig('velxy.png', dpi = 300)
plt.close()

fig2 = plot(p_, title='pressure')
plt.colorbar(fig2)
plt.savefig('pressure.png', dpi = 300)
plt.close()
'''
fig2 = plot(T_, title='temperature')
plt.colorbar(fig2)
plt.savefig('temperature.png', dpi = 300)
plt.close()

fig2 = plot(S_, title='salinity')
plt.colorbar(fig2)
plt.savefig('salinity.png', dpi = 300)
plt.close()

fig2 = plot(div(u_), title='div(u)')
plt.colorbar(fig2)
plt.savefig('divu.png', dpi = 300)
plt.close()

'''
