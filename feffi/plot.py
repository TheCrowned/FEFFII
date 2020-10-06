from . import parameters
from fenics import plot
from pathlib import Path
import logging
import matplotlib.pyplot as plt

def plot_single(to_plot, **kwargs):
    """Plots a single Fenics-plottable object (ex. function, mesh).

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

    # Create plot dir
    Path(config['plot_path']).mkdir(parents=True, exist_ok=True)

    if kwargs.get('title') != None:
        title = kwargs['title']
    elif kwargs.get('file_name') != None:
        title = kwargs['file_name']
    else:
        title = str(to_plot)

    logging.info('Plotting %s...' % title)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pl = plot(to_plot, title = title)

    if config['domain'] == 'fjord':
        ax.set_aspect('auto')

    if kwargs.get('display') != None and kwargs['display'] == True:
        plt.show()
    if kwargs.get('file_name') != None and kwargs['file_name'] != '':
        plt.savefig(config['plot_path'] + kwargs['file_name'], dpi = 800)

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
    See `plot_single`.
    """
    plot_single(f['u_'], file_name='velxy.png', title='Velocity', **kwargs)
    plot_single(f['u_'][0], file_name='velx.png', title='Velocity X-component', **kwargs)
    plot_single(f['u_'][1], file_name='vely.png', title='Velocity Y-component', **kwargs)
    plot_single(f['p_'], file_name='pressure.png', title='Pressure', **kwargs)
    plot_single(f['T_'], file_name='temperature.png', title='Temperature', **kwargs)
    plot_single(f['S_'], file_name='salinity.png', title='Salinity', **kwargs)

    '''fig = plot(div(u_), title='Velocity divergence')
    plt.colorbar(fig)
    plt.savefig(plot_path + 'div_u.png', dpi = 500)
    plt.close()'''

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
