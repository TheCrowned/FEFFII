"""Small test for Navier-Stokes and lid-driven cavity based on Stefano
O's feffi module

"""
import fenics
from fenics import (dot, inner, elem_mult,
                    nabla_grad, div, dx, ds, sym, Identity,
                    DirichletBC, Constant, assemble,
                    solve)

# Physical parameters used by lid-driven-cavity.yml
final_time = 8
steps_n = 800
simulation_precision = -3
g = 0
nu = 1e-2
rho_0 = 1
beta = 0
gamma = 0
T_0 = 0                         # dummy
S_0 = 35                        # dummy

# Mesh
nx = 100
msh = fenics.UnitSquareMesh(nx, nx)

# Function spaces P2/P1
V = fenics.VectorFunctionSpace(msh, 'CG', 2)
Q = fenics.FunctionSpace(msh, 'CG', 1)
TS = fenics.FunctionSpace(msh, 'CG', 2)


# Boundary conditions (semi-manually implemented, so have to check Ste's
# implementation). Only using classes...
class Top(fenics.SubDomain):
    def inside(self, x, on_boundary):
        return fenics.near(x[1], 1) and on_boundary


class No_Slip(fenics.SubDomain):
    def inside(self, x, on_boundary):
        return (
            (fenics.near(x[1], 0) or fenics.near(x[0], 0) or fenics.near(x[0], 1))
            and on_boundary)


top_right_corner = [0, 0]
bcp = DirichletBC(
    Q, Constant(0),
    'near(x[0], {:f}) && near(x[1], {:f})'.format(*top_right_corner),
    method='pointwise'
)
bc_noslip = DirichletBC(V, Constant((0, 0)), No_Slip())
bc_top = DirichletBC(V, Constant((1, 0)), Top())
bcs_v = [bc_noslip, bc_top]

# funcitons, trial functions and test functions
u_n = fenics.Function(V)
u_ = fenics.Function(V)
u = fenics.TrialFunction(V)
v = fenics.TestFunction(V)
p_n = fenics.Function(Q)
p_n_1 = fenics.Function(Q)
p_ = fenics.Function(Q)
p = fenics.TrialFunction(Q)
q = fenics.TestFunction(Q)
T_ = fenics.Function(TS)        # dummy in this case
S_ = fenics.Function(TS)        # dummy in this case

# Assemble tensor viscosity/diffusivity
nu_tensor = fenics.as_tensor((
    (nu, nu),
    (nu, nu)))

# Define expressions used in variational forms
U = 0.5*(u_n + u)
n = fenics.FacetNormal(msh)
dt = final_time/steps_n
gammap = 0.5 # gamma'


def get_matrix_diagonal(mat):
    diag = []
    for i in range(mat.ufl_shape[0]):
        diag.append(mat[i][i])

    return fenics.as_vector(diag)


# Define variational problem for approximated velocity
buoyancy = fenics.Expression(
    (0, '-g*(-beta*(T_ - T_0) + gamma*(S_ - S_0))'),
    beta=beta, gamma=gamma,
    T_0=T_0, S_0=S_0,
    g=g, T_=T_, S_=S_,
    degree=2)

y = fenics.Expression("1-x[1]", domain=msh, degree=2) # DOMAIN ARG!!??

# VARIATIONAL FORMS for 2-eq splitting
F1 = + dot((u - u_n)/dt, v)*dx \
     + dot(dot(u_n, nabla_grad(u)), v)*dx \
     + inner(2*elem_mult(nu_tensor, sym(nabla_grad(u))), sym(nabla_grad(v)))*dx \
     - inner(((1+gammap)*p_n - gammap*p_n_1 - rho_0*g*y)/rho_0*Identity(len(u)), sym(nabla_grad(v)))*dx \
     + dot(((1+gammap)*p_n - gammap*p_n_1 - rho_0*g*y)*n/rho_0, v)*ds \
     - dot(elem_mult(nu_tensor, nabla_grad(u))*n, v)*ds \
     - dot(buoyancy, v)*dx
a1, L1 = fenics.lhs(F1), fenics.rhs(F1)

# Variational problem for pressure p with approximated velocity u
F2 = + dot(nabla_grad(p), nabla_grad(q))/rho_0*dx \
     + div(u_)*q*(1/dt)*dx
a2, L2 = fenics.lhs(F2), fenics.rhs(F2)

# Ste's solution procedure
A1 = assemble(a1)
A2 = assemble(a2)
[bc.apply(A1) for bc in bcs_v]
bcp.apply(A2)

xdmffile = fenics.XDMFFile('lid-driven-cavity-stab/solutions.xdmf')
xdmffile.parameters["flush_output"] = True
xdmffile.parameters["functions_share_mesh"] = True
u_.rename("velocity", "Velocity in m/s")
p_n_1.rename("old pressure", "Old pressure in Pa")
p_.rename("pressure", "Pressure in Pa")
t = 0
no_time_steps = 0
xdmffile.write(u_, t)
xdmffile.write(p_, t)

while no_time_steps <= steps_n:
    # Applying IPC splitting scheme (IPCS)
    # Step 1: Tentative velocity step
    # A1 = self.A1
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcs_v]
    solve(A1, u_.vector(), b1)

    # Step 2: Pressure correction step
    # A2 = self.A2
    b2 = assemble(L2)
    bcp.apply(b2)
    solve(A2, p_.vector(), b2)

    # update solution
    u_n.assign(u_)
    p_n_1.assign(p_n)
    p_n.assign(p_)
    t += dt
    if not no_time_steps % 10:
        xdmffile.write(u_, t)
        xdmffile.write(p_n_1, t)
        xdmffile.write(p_, t)
    no_time_steps += 1
    print("Timestep: {}/{}".format(no_time_steps, steps_n))


# naive implementation: Christian
domain = fenics.UnitSquareMesh(100, 100)
n = fenics.FacetNormal(domain)
V_ = fenics.VectorElement("Lagrange", domain.ufl_cell(), 2)
P_ = fenics.FiniteElement("Lagrange", domain.ufl_cell(), 1)
W = fenics.FunctionSpace(domain, V_ * P_)
Re = Constant(10**2)

w_trial = fenics.TrialFunction(W)
w_test = fenics.TestFunction(W)
(u, p) = fenics.split(w_trial)
(v, q) = fenics.split(w_test)
sol = fenics.Function(W)
bcp_ = DirichletBC(
    W.sub(1), Constant(0),
    'near(x[0], {:f}) && near(x[1], {:f})'.format(*top_right_corner),
    method='pointwise'
)
bc_noslip_ = DirichletBC(W.sub(0), Constant((0, 0)), No_Slip())
bc_top_ = DirichletBC(W.sub(0), Constant((1, 0)), Top())
bcs = [bcp, bc_top, bc_noslip]
solver_params = {
    'linear_solver': 'mumps',
    'preconditioner': 'default',
    'maximum_iterations': 10,
    'relaxation_parameter': 1.0,
    'relative_tolerance': 1e-7,
}


# Define strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))


# Define stress tensor
def sigma(u, p, nu):
    return 2*(nu*epsilon(u)) - p*Identity(len(u))


def sigma_elem(u, p, nu_tensor):
    return 2*elem_mult(nu_tensor, epsilon(u)) - p*Identity(len(u))

steady_formulation = (
    + dot(dot(u, nabla_grad(u)), v)*dx
    + nu*inner(nabla_grad(u), nabla_grad(v))*dx
    - dot(p, fenics.div(v))*dx
    - dot(fenics.div(u), q)*dx)

steady_formulation_stress = (
    + dot(dot(u, nabla_grad(u)), v)*dx
    + inner(sigma_elem(u, p, nu_tensor), epsilon(v))*dx
    - dot(fenics.div(u), q)*dx)
F_vel = fenics.action(steady_formulation, sol)
F_stress = fenics.action(steady_formulation_stress, sol)
solve(F_stress == 0,
      sol, bcs=[bcp_, bc_noslip_, bc_top_],
      solver_parameters={'newton_solver': solver_params}
)

(u_sol, p_sol) = sol.split(True)

xdmff = fenics.XDMFFile('lid-driven-cavity/steady_solutions.xdmf')
xdmff.parameters["flush_output"] = True
xdmff.parameters["functions_share_mesh"] = True
u_sol.rename("velocity", "Velocity in m/s")
p_sol.rename("pressure", "Pressure in Pa")
xdmff.write(u_sol, 0)
xdmff.write(p_sol, 0)

err = fenics.errornorm(u_, u_sol, norm_type='L2')
err_p = fenics.errornorm(p_, p_sol, norm_type='L2')
print("||u_{time} - u_{steady}||_{L^2}: ", err)
print("||p_{time} - p_{steady}||_{L^2}: ", err_p)

