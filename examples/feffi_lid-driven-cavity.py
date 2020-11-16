# feffi module lives one level up current dir
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from feffi import *
import fenics
import matplotlib.pyplot as plt

parameters.define_parameters({
    'config_file' : 'feffi/config/lid-driven-cavity.yml',
    #'max_iter' : 20
})
parameters.parse_commandline_args()

mesh = mesh.create_mesh()
f_spaces = functions.define_function_spaces(mesh)
f = functions.define_functions(f_spaces)
#functions.init_functions(f) # Init functions to closest steady state
(stiffness_mats, load_vectors) = functions.define_variational_problems(f, mesh)
domain = boundaries.Domain(mesh, f_spaces)

flog.info(
    '## Running lid driven benchmark with parameters \n{} ##'.format(
        parameters.config))

simul = simulation.Simulation(f, stiffness_mats, load_vectors, domain.BCs)
simul.run()

plot.plot_single(
    f['u_'],
    title='Velocity (nu = {})'.format(parameters.config['nu']),
    display=True)
plot.plot_single(
    f['p_'],
    title='Pressure (nu = {})'.format(parameters.config['nu']),
    display=True)

flog.info('Moving log file to plot folder')
system('mv simulation.log "' + parameters.config['plot_path'] + '/simulation.log"')

# Export solutions for comparison
#fenics.File('lid-driven-cavity_u_{}.xml'.format(parameters.config['nu'])) << f['u_']
#fenics.File('lid-driven-cavity_p_{}.xml'.format(parameters.config['nu'])) << f['p_']
#precision reached at 6954 step

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

plt.show()