# feffi module lives one level up current dir
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from feffi import *
import fenics
from fenics import SubDomain, dot, nabla_grad, grad, div, dx
import matplotlib.pyplot as plt

parameters.define_parameters({
    'config_file' : 'feffi/config/lid-driven-cavity.yml',
    'max_iter' : 1000
})
parameters.parse_commandline_args()


# Uncomment this to use the COMSOL-like mesh, with rough interior and
# progressively finer boundaries
'''mesh = fenics.UnitSquareMesh(30,30)

# Multiple passes of refinement from different distances from boundaries
for thresh in [0.2, 0.1, 0.05, 0.025]:
    class Bound_Top(SubDomain):
        def inside(self, x, on_boundary):
            return x[1] >= 1-thresh and x[0] <= 1-thresh
    class Bound_Right(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] >= 1-thresh and x[1] >= thresh
    class Bound_Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return x[1] <= thresh and x[0] >= thresh
    class Bound_Left(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] <= thresh and x[1] <= 1-thresh

    boundary_domain = fenics.MeshFunction("bool", mesh, mesh.topology().dim() - 1)
    boundary_domain.set_all(False)

    boundariess = [Bound_Top, Bound_Right, Bound_Bottom, Bound_Left]
    for boundary in boundariess:
        obj = boundary()
        obj.mark(boundary_domain, True)

    mesh = fenics.refine(mesh, boundary_domain)'''

mesh = mesh.create_mesh()

f_spaces = functions.define_function_spaces(mesh)
f = functions.define_functions(f_spaces)
#functions.init_functions(f) # Init functions to closest steady state
(stiffness_mats, load_vectors) = functions.define_variational_problems(f, mesh)

## Apply CBS stabilization
nu = parameters.assemble_viscosity_tensor(parameters.config['nu']);

gamma = 1
dt = 1/parameters.config['steps_n']
tau = (2*gamma - 1)*dt/2
F = fenics.Constant((0,0)) # g is null for lid driven cavity

def R():
    return dot(f['u_n'], nabla_grad(f['u_n'])) - nu*div(nabla_grad(f['u_n'])) + grad(f['p_n']) - F

# Rebuild complete weak form and add stabilization term
F1 = stiffness_mats['a1'] + load_vectors['L1']
F1 += tau*dot(dot(f['u_n'], grad(f['v'])), R())*dx

load_vectors['L1'] = fenics.rhs(F1)
stiffness_mats['a1'] = fenics.lhs(F1)

# Trying to see how much stab affects - no success
#fenics.norm(delta*dot(grad(u_), R(u_))))
#fenics.plot(delta*dot(grad(u_), R(u_))))
#a = fenics.Expression('delta*dot(grad(u), R(u))', degree=2, delta=delta, u = f['u_'])
#flog.info(fenics.norm(a))
#plot(a)
#plt.show()
## End of CBS stabilization

domain = boundaries.Domain(mesh, f_spaces)

flog.info(
    '## Running lid driven benchmark with parameters \n{} ##'.format(
        parameters.config))

simul = simulation.Simulation(f, stiffness_mats, load_vectors, domain.BCs)
simul.run()


plot.plot_single(
    mesh,
    title='Mesh (nu = {})'.format(parameters.config['nu']),
    display=False,
    file_name='mesh-{}.png'.format(parameters.config['nu']),
    )
plot.plot_single(
    f['u_'],
    title='Velocity (nu = {})'.format(parameters.config['nu']),
    display=False,
    file_name='vel-{}.png'.format(parameters.config['nu']),
    )
plot.plot_single(
    f['p_'],
    title='Pressure (nu = {})'.format(parameters.config['nu']),
    display=False,
    file_name='pressure-{}.png'.format(parameters.config['nu']),
    )

# Export solutions for comparison
#fenics.File('out/lid-driven-cavity_u_{}.xml'.format(parameters.config['nu'])) << f['u_']
#fenics.File('out/lid-driven-cavity_p_{}.xml'.format(parameters.config['nu'])) << f['p_']

# Compare with reference solutions
ref_u = fenics.Function(f_spaces['V'])
ref_p = fenics.Function(f_spaces['Q'])
fenics.File(
    'feffi/reference-solutions/lid-driven-cavity_u_{}.xml'.format(
        parameters.config['nu'])) >> ref_u
fenics.File(
    'feffi/reference-solutions/lid-driven-cavity_p_{}.xml'.format(
        parameters.config['nu'])) >> ref_p

fig = plt.figure()
ax = fig.add_subplot(2,3,1)
pl = fenics.plot(f['u_'], title = 'Current u')
plt.colorbar(pl)
ax = fig.add_subplot(2,3,2)
pl = fenics.plot(ref_u, title = 'Ref. u')
plt.colorbar(pl)
ax = fig.add_subplot(2,3,3)
pl = fenics.plot(f['u_']-ref_u, title = 'Current - ref')
plt.colorbar(pl)
ax = fig.add_subplot(2,3,4)
pl = fenics.plot(f['p_'], title = 'Current p')
plt.colorbar(pl)
ax = fig.add_subplot(2,3,5)
pl = fenics.plot(ref_p, title = 'Ref. p')
plt.colorbar(pl)
ax = fig.add_subplot(2,3,6)
pl = fenics.plot(f['p_']-ref_p, title = 'Current - ref')
plt.colorbar(pl)

plt.savefig('out/comp.png', dpi=800)
plt.show()
