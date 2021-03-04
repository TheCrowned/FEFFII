"""Steady state Navier-Stokes lid-driven cavity with
the addition of some stabilization.
"""
import fenics
import numpy as np
import matplotlib.pyplot as plt
from fenics import (dot, inner, elem_mult, grad,
                    nabla_grad, div, dx, ds, sym, Identity,
                    DirichletBC, Constant, assemble,
                    solve)

# ------------------
# Setting parameters
# ------------------
Re = 1000
nu = 1/Re
stab = True
nofpoints = 50  # for mesh
tol = 1e-10  #for steady state iterations
max_iter = 35
l = 2 # velocity elements degree
k = 1 # pressure elements degree

print("--------------------------")
print("Reynolds number: "+str(Re))
print("--------------------------")

# ------------------------
# Creating FEM ingredients
# ------------------------

# mesh and elements
domain = fenics.UnitSquareMesh(nofpoints, nofpoints)
hmin = domain.hmin(); hmax = domain.hmax()
n = fenics.FacetNormal(domain)
V_ = fenics.VectorElement("Lagrange", domain.ufl_cell(), l)
P_ = fenics.FiniteElement("Lagrange", domain.ufl_cell(), k)

# test and trial functions
W = fenics.FunctionSpace(domain, V_ * P_)
w_trial = fenics.TrialFunction(W)
w_test = fenics.TestFunction(W)
(u, p) = fenics.split(w_trial)
(v, q) = fenics.split(w_test)
sol = fenics.Function(W)

# --------------------------
# Setting boundaries and BCs
# --------------------------

class Top(fenics.SubDomain):
    def inside(self, x, on_boundary):
        return fenics.near(x[1], 1) and on_boundary

class No_Slip(fenics.SubDomain):
    def inside(self, x, on_boundary):
        return (
            (fenics.near(x[1], 0) or fenics.near(x[0], 0) or fenics.near(x[0], 1))
            and on_boundary)

bottom_left_corner = [0, 0]

bcp_ = DirichletBC(
    W.sub(1), Constant(0),
    'near(x[0], {:f}) && near(x[1], {:f})'.format(*bottom_left_corner),
    method='pointwise'
)
bc_noslip_ = DirichletBC(W.sub(0), Constant((0, 0)), No_Slip())
bc_top_ = DirichletBC(W.sub(0), Constant((1, 0)), Top())

# ------------------------
# Defining forms
# ------------------------

def N(a, u, p):
    return -nu*div(nabla_grad(u)) + dot(a, nabla_grad(u)) + grad(p)

def B_g(a, u, p, v, q):
    return (
    + nu*inner(nabla_grad(u), nabla_grad(v))*dx
    + (1/2)*(dot(dot(a, nabla_grad(u)), v) - dot(dot(a, nabla_grad(v)), u))*dx # why a minus? Obscure.
    - dot(p, div(v))*dx
    - dot(div(u), q)*dx ) # this is the only term that is not strictly in B_g in the paper


# Init some vars
n = 0
residual = 1e22
plot_info = {'Rej': [], 'norm_a': [], 'delta': [], 'tau': [], 'residual': []}

# Start the dance
while residual > tol and n <= max_iter:

    a = sol.split(True)[0]

    # ------------------------
    # Setting stab. parameters
    # ------------------------

    norm_a = fenics.norm(a)
    if norm_a == 0: #first iteration, a = 0 -> would div by zero
       norm_a = 1

    Rej = norm_a*hmin/(2*nu)
    delta0 = 0.1 # "tuning parameter" > 0
    tau0 = 0.2 if l == 2 else 0 # "tuning parameter" > 0 dependent on V.degree
    delta = delta0*hmin*min(1, Rej/3)/norm_a
    tau = tau0*max(nu, hmin)
    alpha = hmax**(1+max(k, l))

    print('n = {}; Rej = {}; delta = {}; tau = {}; alpha = {}'.format(
        n, round(Rej, 5), round(delta, 5), round(tau, 5), round(alpha, 5)))

    # ------------------------------
    # Define variational formulation
    # ------------------------------

    steady_form = B_g(a, u, p, v, q) + alpha*dot(p, q)*dx
    if stab:
        #turn individual terms on and off by tweaking delta0, tau0
        if delta > 0:
            steady_form += delta*(dot(N(a, u, p), N(a, v, q)))*dx
        if tau > 0:
            steady_form += tau*(dot(div(u), div(v)))*dx

    solve(fenics.lhs(steady_form) == fenics.rhs(steady_form),
          sol, bcs=[bcp_, bc_noslip_, bc_top_])

    u_new = sol.split(True)[0]
    residual = fenics.errornorm(u_new, a)
    print(" >>> residual is " + str(residual) + "<<<")

    # Save info to plot later
    plot_info['residual'].append(residual)
    plot_info['Rej'].append(Rej)
    plot_info['tau'].append(tau)
    plot_info['delta'].append(delta)
    plot_info['norm_a'].append(norm_a)

    n += 1

(u_sol, p_sol) = sol.split(True)

x = list(range(n))
plt.plot(x, plot_info['residual'], label = "Residual")
plt.plot(x, plot_info['Rej'], label = "Rej")
plt.plot(x, plot_info['tau'], label = "tau")
plt.plot(x, plot_info['delta'], label = "delta")
plt.plot(x, plot_info['norm_a'], label = "norm_a")
plt.legend()
plt.show()

pl=fenics.plot(u_sol)
plt.colorbar(pl)
plt.show()
plt.close()
pl=fenics.plot(p_sol)
plt.colorbar(pl)
plt.show()
