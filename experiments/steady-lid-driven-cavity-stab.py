"""Small test for Navier-Stokes and lid-driven cavity based on Stefano
O's feffi module, with the addition of CBS stabilization.
"""
import fenics
import matplotlib.pyplot as plt
from fenics import (dot, inner, elem_mult, grad,
                    nabla_grad, div, dx, ds, sym, Identity,
                    DirichletBC, Constant, assemble,
                    solve)

output_dir = 'lid-driven-cavity-stab/'

# Physical parameters
nu = 1e-3

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

bottom_left_corner = [0, 0]

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
    'near(x[0], {:f}) && near(x[1], {:f})'.format(*bottom_left_corner),
    method='pointwise'
)
bc_noslip_ = DirichletBC(W.sub(0), Constant((0, 0)), No_Slip())
bc_top_ = DirichletBC(W.sub(0), Constant((1, 0)), Top())
solver_params = {
    'linear_solver': 'mumps',
    'preconditioner': 'default',
    'maximum_iterations': 10,
    'relaxation_parameter': 1.0,
    'relative_tolerance': 1e-7,
}

print(domain.hmin())
print(domain.hmin()**2/nu)

tau0 = 1
delta = domain.hmin()**2/nu
tau = tau0*max(domain.hmin(), nu)

def N(v,p):
    return -nu*div(nabla_grad(v)) + dot(u, nabla_grad(v)) + grad(p)

steady_formulation = (
    + dot(dot(u, nabla_grad(u)), v)*dx
    + nu*inner(nabla_grad(u), nabla_grad(v))*dx
    - dot(p, div(v))*dx
    - dot(div(u), q)*dx
    + delta*(dot(N(u,p), N(v,q)))*dx
    + tau*(dot(div(u), div(v)))*dx
    )
F_vel = fenics.action(steady_formulation, sol)
solve(F_vel == 0,
      sol, bcs=[bcp_, bc_noslip_, bc_top_],
      solver_parameters={'newton_solver': solver_params}
)

(u_sol, p_sol) = sol.split(True)

pl=fenics.plot(u_sol)
plt.colorbar(pl)
plt.show()
plt.close()
pl=fenics.plot(p_sol)
plt.colorbar(pl)
plt.show()