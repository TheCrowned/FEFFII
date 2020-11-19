from fenics import *
import matplotlib.pyplot as plt

'''
See https://quickmathintuitions.org/reproducing-transport-instability-convection-diffusion-equation/
The instability shows up with epsilon = 0.01 and mesh h = 0.1 for example (set beta to 0).
However, if we add an artificial viscosity coefficient, the issue can vanish. Try beta = 0.5 for ex.
This ensures the term with epsilon always has a numerator not smaller than one order of magnitude than the denominator.
'''

mesh = UnitIntervalMesh(35)
V = FunctionSpace(mesh, 'P', 1)
bcu = [
    DirichletBC(V, Constant(0), 'near(x[0], 0)'),
    DirichletBC(V, Constant(0), 'near(x[0], 1)'),
]
u = TrialFunction(V)
v = TestFunction(V)
u_  = Function(V)
f  = Constant(1)
epsilon  = Constant(0.01)
beta = Constant(0.3)
hmin = mesh.hmin()

a = (epsilon+beta*hmin)*dot(u.dx(0), v.dx(0))*dx + u.dx(0)*v*dx
L = v*dx

solve(a == L, u_, bcs=bcu)

print("||u|| = %s, ||u||_8 = %s" % ( \
    round(norm(u_, 'L2'), 2), round(norm(u_.vector(), 'linf'), 3)
))
#print(u_.vector().get_local())

fig2 = plt.scatter(mesh.coordinates(), u_.compute_vertex_values())
plt.savefig('velxy.png', dpi = 300)
plt.close()

#plot(mesh)
#plt.show()