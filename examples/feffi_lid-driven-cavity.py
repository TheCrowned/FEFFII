# feffi module lives one level up current dir
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from feffi import *
import logging
import fenics
from fenics import SubDomain
import matplotlib.pyplot as plt

parameters.define_parameters({
    'config_file' : 'feffi/config/lid-driven-cavity.yml',
    'max_iter' : 1000
})
parameters.parse_commandline_args()

## my custom mesh

mesh = fenics.UnitSquareMesh(30,30)

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

    mesh = fenics.refine(mesh, boundary_domain)


f_spaces = functions.define_function_spaces(mesh)
f = functions.define_functions(f_spaces)
#functions.init_functions(f) # Init functions to closest steady state
(stiffness_mats, load_vectors) = functions.define_variational_problems(f, mesh)
domain = boundaries.Domain(mesh, f_spaces)

logging.info(
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
fenics.File('out/lid-driven-cavity_u_{}.xml'.format(parameters.config['nu'])) << f['u_']
fenics.File('out/lid-driven-cavity_p_{}.xml'.format(parameters.config['nu'])) << f['p_']

# Compare with reference solutions
'''ref_u = fenics.Function(f_spaces['V'])
ref_p = fenics.Function(f_spaces['Q'])
fenics.File(
    'lid-driven-cavity_u_{}.xml'.format(
        parameters.config['nu'])) >> ref_u
fenics.File(
    'lid-driven-cavity_p_{}.xml'.format(
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

plt.savefig('out/comp.png', dpi=800)'''
