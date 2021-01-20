"""Small test for Navier-Stokes and lid-driven cavity based on Stefano
O's feffi module, with the addition of CBS stabilization.

"""
import fenics
import matplotlib.pyplot as plt
from fenics import (dot, inner, elem_mult, grad,
                    nabla_grad, div, dx, ds, sym, Identity,
                    DirichletBC, Constant, assemble,
                    solve)

# Physical parameters used by lid-driven-cavity.yml
final_time = 3
steps_n = 300
g = 0
nu = 5e-3

# Mesh
nx = 100
msh = fenics.UnitSquareMesh(nx, nx)

# Function spaces P2/P1
V = fenics.VectorFunctionSpace(msh, 'CG', 2)
Q = fenics.FunctionSpace(msh, 'CG', 1)

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
u = fenics.Function(V)
v = fenics.TestFunction(V)
p_n = fenics.Function(Q)
p_n_1 = fenics.Function(Q)
p_ = fenics.Function(Q)
p = fenics.TrialFunction(Q)
q = fenics.TestFunction(Q)

# Define expressions used in variational forms
U = 0.5*(u_n + u)
n = fenics.FacetNormal(msh)
dt = final_time/steps_n
gammap = 0.1 # gamma' #I tried 0.1, 0.15, 0.2, 0.5
tau_s = dt*(1-gammap)
tau_cg = dt/2

# VARIATIONAL FORMS for 2-eq splitting
F1 = + dot((u - u_n)/dt, v)*dx \
     + dot(dot(u, nabla_grad(u)), v)*dx \
     + inner(2*nu*sym(nabla_grad(u)), sym(nabla_grad(v)))*dx \
     - inner(((1+gammap)*p_n - gammap*p_n_1)*Identity(len(u)), sym(nabla_grad(v)))*dx \
     + dot(((1+gammap)*p_n - gammap*p_n_1)*n, v)*ds \
     - dot(nu*nabla_grad(u)*n, v)*ds \
     - tau_cg*dot(dot(u_n, nabla_grad(v)), dot(u_n, nabla_grad(u_n)) - nu*u_n + grad(p_n))*dx

# Variational problem for pressure p with approximated velocity u
F2 = + div(u)*q*dx \
     + tau_s*dot(nabla_grad(p), nabla_grad(q))*dx

A2 = assemble(fenics.lhs(F2))

xdmffile = fenics.XDMFFile('lid-driven-cavity-stab/solutions.xdmf')
xdmffile.parameters["flush_output"] = True
xdmffile.parameters["functions_share_mesh"] = True
u.rename("velocity", "Velocity in m/s")
p_.rename("pressure", "Pressure in Pa")
p_n_1.rename("old pressure", "Old pressure in Pa")
t = 0
no_time_steps = 0
xdmffile.write(u, t)
xdmffile.write(p_, t)
xdmffile.write(p_n_1, t)

solver_params = {'absolute_tolerance': 1e-4, 'maximum_iterations': 10}

while no_time_steps <= steps_n:
    # Applying IPC splitting scheme (IPCS)
    # Step 1: Tentative velocity step
    # A1 = self.A1
    #b1 = assemble(L1)
    #[bc.apply(b1) for bc in bcs_v]
    #solve(A1, u_.vector(), b1)
    solve(F1==0, u, bcs=bcs_v, solver_parameters={'newton_solver': solver_params})

    # Step 2: Pressure correction step
    # A2 = self.A2
    b2 = assemble(fenics.rhs(F2))
    bcp.apply(b2)
    solve(A2, p_.vector(), b2)

    # update solution
    u_n.assign(u)
    p_n_1.assign(p_n)
    p_n.assign(p_)
    t += dt
    if not no_time_steps % 1:
        xdmffile.write(u, t)
        xdmffile.write(p_n_1, t)
        xdmffile.write(p_, t)
    no_time_steps += 1
    print("Timestep: {}/{}".format(no_time_steps, steps_n))

u_old = u

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

# Assemble tensor viscosity/diffusivity
nu_tensor = fenics.as_tensor((
    (nu, nu),
    (nu, nu)))

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

xdmff = fenics.XDMFFile('lid-driven-cavity-stab/steady_solutions.xdmf')
xdmff.parameters["flush_output"] = True
xdmff.parameters["functions_share_mesh"] = True
u_sol.rename("velocity", "Velocity in m/s")
p_sol.rename("pressure", "Pressure in Pa")
xdmff.write(u_sol, 0)
xdmff.write(p_sol, 0)

err = fenics.errornorm(u_old, u_sol, norm_type='L2')
err_p = fenics.errornorm(p_, p_sol, norm_type='L2')
print("||u_{time} - u_{steady}||_{L^2}: ", err)
print("||p_{time} - p_{steady}||_{L^2}: ", err_p)