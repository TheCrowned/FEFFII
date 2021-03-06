from . import parameters
from fenics import plot, norm
import fenics
import matplotlib.pyplot as plt
import logging
import os

flog = logging.getLogger('feffii')


def plot_single(to_plot, **kwargs):
    """Plots a single Fenics-plottable object (ex. function, mesh).

    Warning: 3D plotting only works if `ax.set_aspect('equal')` is disabled
    in FEniCS (https://bitbucket.org/fenics-project/dolfin/issues/1087/3d-objects-cannot-be-plotted-with).

    Parameters
    ----------
    to_plot : fenics-plottable object

    kwargs
    ------
    title : str
        plot title. If nothing is given, we'll make up something.
    file_name : str
        plot file name. If given, plot will be saved in `config.plot_path`
        as a png file.
    display : bool (default = False)
        whether to display plot.
    plot_path : str
        where to plot (a default directory is set up in the `plots` folder).
    """

    # Allow function arguments to overwrite wide config (but keep it local)
    config = dict(parameters.config); config.update(kwargs)

    if kwargs.get('title') != None:
        title = kwargs['title']
    elif kwargs.get('file_name') != None:
        title = kwargs['file_name']
    else:
        title = str(to_plot)

    flog.info('Plotting %s...' % title)

    # Plot size is aspect-ratio * 2 (+2 on x to fit colorbar)
    try: # if to_plot is a Function
        mesh = to_plot.function_space().mesh()
        figsize = (max(mesh.coordinates()[:,0]) + 2,
                   max(mesh.coordinates()[:,1]))
    except AttributeError: # to_plot is then probably a mesh
        try:
            figsize = (max(to_plot.coordinates()[:,0]) + 2,
                       max(to_plot.coordinates()[:,1]))
        except: # plotting rocks
            figsize = (12, 10)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    pl = plot(to_plot, title=title)
    ax.set_aspect('auto')  # forces square plot

    # Add colorbar if possible (i.e. if it is a Function)
    try:
        plt.colorbar(pl)#, cax = fig.add_axes([0.92, 0.2, 0.03, 0.55]))
    except:
        pass

    if config['domain'] == 'fjord':
        ax.set_aspect('auto')

    if kwargs.get('display') != None and kwargs['display'] == True:
        plt.show()
    if kwargs.get('file_name') != None and kwargs['file_name'] != '':
        plt.savefig(os.path.join(
            config['plot_path'], kwargs['file_name']), bbox_inches='tight', pad_inches=0.1, dpi=400)

    plt.close()


def plot_solutions(f, **kwargs):
    """Plots all solution functions.

    Parameters
    ----------
    f : dict
        Fenics functions to plot.
        Should contain entries `u_`, `p_`, `T_`, `S_`.

    kwargs
    ------
    All of `plot_single`.
    """

    plot_single(f['u_'].function_space().mesh(), file_name='mesh.png',
                title='Mesh', **kwargs)
    plot_single(f['u_'], file_name='velxy.png',
                title='Velocity', **kwargs)
    #plot_single(fenics.div(f['u_']), file_name='div_velxy.png',
    #            title='div(Velocity)', **kwargs)
    # Velocity components
    for i in range(f['u_'].geometric_dimension()):
        plot_single(f['u_'].sub(i), file_name='velx{}.png'.format(i),
                    title='Velocity component {}'.format(i), **kwargs)
    plot_single(f['p_'], file_name='pressure.png',
                title='Pressure', **kwargs)
    plot_single(f['T_'], file_name='temperature.png',
                title='Temperature', **kwargs)
    plot_single(f['S_'], file_name='salinity.png',
                title='Salinity', **kwargs)

    melt_sol = f['3eqs']['sol'].split()
    #plot_single(f['3eqs']['uStar'], file_name='uStar.png', title='uStar', **kwargs)
    #plot_single(melt_sol[0], file_name='m_B.png', title='m_B', **kwargs)
    #plot_single(melt_sol[1], file_name='T_B.png', title='T_B', **kwargs)
    #plot_single(melt_sol[2], file_name='S_B.png', title='S_B', **kwargs)

    '''bmesh = BoundaryMesh(mesh, "exterior", True)
    boundary = bmesh.coordinates()
    BC = { 'x': [], 'y': [], 'ux': [], 'uy': [] }
    fig = plt.figure()
    for i in range(len(boundary)):
        BC['x'].append(boundary[i][0])
        BC['y'].append(boundary[i][1])
        BC['ux'].append(u_(boundary[i])[0])
        BC['uy'].append(u_(boundary[i])[1])
    plt.quiver(BC['x'], BC['y'], BC['ux'], BC['uy'])
    plt.savefig(plot_path + 'boundaryvel.png', dpi = 500)
    plt.close()'''

'''def plot_boundary_conditions():
    _u_1, _u_2 = u_.split(True)

    bmesh = BoundaryMesh(mesh, "exterior", True)
    boundarycoords = bmesh.coordinates()

    rightboundary=[]
    xvelrightboundary=[]
    BC = np.empty(shape=(len(boundarycoords), 2)) #https://stackoverflow.com/a/569063
    for i in range(len(boundarycoords)):

        if boundarycoords[i][0]==1:
            BC[i][1] = boundarycoords[i][1]
            BC[i][0] = (_u_1(boundarycoords[i]))/10+1
        else:
            BC[i][0] = boundarycoords[i][0]
            BC[i][1] = boundarycoords[i][1]

    fig6 = plt.figure()
    plt.scatter(BC[:,0], BC[:,1])
    plt.savefig('boundary_conditions.png', dpi = 300)
    plt.close()

    #plt.scatter(boundarycoords[:,0], boundarycoords[:,1])
    '''
