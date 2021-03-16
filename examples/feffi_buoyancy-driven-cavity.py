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
    'config_file' : 'feffi/config/buoyancy-driven-cavity.yml',
    'max_iter' : 3000
})
parameters.parse_commandline_args()

mesh = mesh.create_mesh()
f_spaces = functions.define_function_spaces(mesh)
f = functions.define_functions(f_spaces)
functions.init_functions(f) # Init functions to closest steady state
domain = boundaries.Domain(mesh, f_spaces)

logging.info(
    '## Running buoyancy driven benchmark with parameters \n{} ##'.format(
        parameters.config))

simul = simulation.Simulation(f, domain.BCs)
simul.run()

plot.plot_single(
    f['u_'],
    title='Velocity (Ra = {})'.format(parameters.config['beta']),
    display=True)
plot.plot_single(
    f['u_'].sub(0),
    title='X-Velocity (Ra = {})'.format(parameters.config['beta']),
    display=True)
'''plot.plot_single(
    f['p_'],
    title='Pressure (Ra = {})'.format(parameters.config['beta']),
    display=True)'''
plot.plot_single(
    f['T_'],
    title='Temperature (Ra = {})'.format(parameters.config['beta']),
    display=True)

flog.info('Moving log file to plot folder')
system('mv simulation.log "' + parameters.config['plot_path'] + '/simulation.log"')

# Export solutions for comparison
#fenics.File('buoyancy-driven-cavity_u_{}.xml'.format(parameters.config['beta'])) << f['u_']
#fenics.File('buoyancy-driven-cavity_p_{}.xml'.format(parameters.config['beta'])) << f['p_']
#fenics.File('buoyancy-driven-cavity_T_{}.xml'.format(parameters.config['beta'])) << f['T_']

# Compare with reference solutions
'''ref_u = fenics.Function(f_spaces['V'])
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

plt.show()'''