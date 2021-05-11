from fenics import *
import numpy as np
import matplotlib.pyplot as plt

'''
See https://quickmathintuitions.org/reproducing-transport-instability-convection-diffusion-equation/
The instability shows up with epsilon = 0.01 and mesh nx = 10 (so that h = 0.1) for example.
'''

dim = 2
nx = 10
epsilon  = 0.01
delta = 0.05
apply_GLS_stab = True

if dim == 1:
    mesh = UnitIntervalMesh(nx)
elif dim == 2:
    mesh = UnitSquareMesh(nx, nx)

if apply_GLS_stab:
    print('Running WITH stabilization')
    output_name = 'vel_stab.png'
else:
    print('Running withOUT stabilization')
    output_name = 'vel_nonstab.png'

V = FunctionSpace(mesh, 'P', 1) #why having 2 here makes the problem vanish even with h = 0.1, epsilon = 0.01?? THINK about what is the role of the degree of functionspace wrt solution. How does it affect it? Visualize it.

bcu = DirichletBC(V, Constant(0), 'on_boundary') #BC applied to whole boundary. This is what makes the solution illy constrained

u = TrialFunction(V)
v = TestFunction(V)
u_ = Function(V)
b_vector = (1,)*dim #the comma forces a list of size 1. Specifying 'Constant' is only needed when UFL can't cast it by itself
b = Constant(b_vector) #the comma forces a list of size 1. Specifying 'Constant' is only needed when UFL can't cast it by itself
f = Constant(1) #but is also ok when we don't need to easily access the value

# define stabilization parameter delta value with formula according to Bengzon Larsson book, section 10.1.6.
C = 0.5 #this is pretty random. According to Chris it is a standard value in FEM, but not mention at all in Larsson-Bengzon book
hmin = mesh.hmin()
delta = C*hmin**2 if epsilon > hmin else C*hmin/np.linalg.norm(b_vector, np.inf)
print("Delta {}".format(delta))

a = epsilon*dot(grad(u), grad(v))*dx + dot(b, grad(u))*v*dx
L = dot(f, v)*dx

def L_operator(u):
    '''GLS differential operator L'''

    return -epsilon*div(grad(u)) + dot(b, grad(u))

if apply_GLS_stab:
    a += delta*dot(L_operator(u), L_operator(v))*dx
    L += delta*dot(f, L_operator(v))*dx

solve(a == L, u_, bcs=bcu)

print("||u|| = %s, ||u||_8 = %s" % ( \
    round(norm(u_, 'L2'), 2), round(norm(u_.vector(), 'linf'), 3)
))

fig = plot(u_)
plt.colorbar(fig)
plt.show()