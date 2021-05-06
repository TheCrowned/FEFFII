import fenics
import logging
from feffi.shelfgeometry import ShelfGeometry
import numpy as np
from math import sqrt
from . import parameters

flog = logging.getLogger('feffi')


def create_mesh(**kwargs):
    """Generates domain and mesh.

    Relies on global config, but parameters can be passed overwriting the globals.

    Parameters
    ----------
    kwargs : `domain`, `mesh_resolution`, `domain_size_x`, `domain_size_y`,
    `shelf_size_x`, `shelf_size_y`, `mesh_resolution_x`, `mesh_resolution_y`,
    `mesh_resolution_sea_y` (refer to README for info).

    Examples
    --------
    1) Generate mesh on 1x1 square domain, with mesh resolution 15x15:

       feffi.mesh(domain = 'square', mesh_resolution = 15)

    1) Generate mesh on 3x2 domain, using globals for all other configs:

       feffi.mesh(domain = 'custom', domain_size_x = 3, domain_size_y = 2)

    Return
    ------
    mesh : fenics mesh.
    """

    # Allow function arguments to overwrite wide config (but keep it local)
    config = dict(parameters.config); config.update(kwargs)

    if config['domain'] == "square":
        mesh = fenics.UnitSquareMesh(
            config['mesh_resolution'], config['mesh_resolution'])

    if config['domain'] == "fjord":
        # general domain geometry: width, height, ice shelf width, ice shelf thickness
        domain_params = [config['domain_size_x'], config['domain_size_y'],
                         config['shelf_size_x'], config['shelf_size_y']]

        sg = ShelfGeometry(
            domain_params,
            ny_ocean = config['mesh_resolution_y'],      # layers on "deep ocean" (y-dir)
            ny_shelf = config['mesh_resolution_sea_y'],  # layers on "ice-shelf thickness" (y-dir)
            nx = config['mesh_resolution_x'],            # layers x-dir
        )

        sg.generate_mesh()
        mesh = sg.get_fenics_mesh()

        #self.refine_mesh_at_point(Point(self.args.shelf_size_x, self.args.domain_size_y - self.args.shelf_size_y))

        '''fenics_domain = Rectangle(Point(0., 0.), Point(1., 1.)) - \
                        Rectangle(Point(0.0, 0.9), Point(0.4, 1.0))
        mesh = generate_mesh(fenics_domain, resolution, "cgal")
        deform_mesh_coords(mesh)

        mesh = refine_mesh_at_point(mesh, Point(0.4, 0.9), domain)'''

    logging.info('Initialized mesh: vertexes {}, hmax {:.2f}, hmin {:.2f}'
                 .format(mesh.num_vertices(), mesh.hmax(), mesh.hmin()))

    return mesh


def add_sill(mesh, center, height, width):
    """Performs (in-place) mesh deformation to create a bottom sill.

    Run this *after* domains have been marked with
    `boundaries.Domain.mark_boundaries()`, otherwise the sill countour will
    not have proper BCs applied.

    Parameters
    ----------
    mesh : Mesh to deform
    center : (float) center x-coordinate of sill
    height : (float) height of the tip
    width : (float) width x-wise of sill

    Examples
    --------
    1) mesh = feffi.mesh.create_mesh()
       f_spaces = feffi.functions.define_function_spaces(mesh)
       domain = feffi.boundaries.Domain(mesh, f_spaces)
       feffi.mesh.add_sill(mesh, 30, 0.8, 25)
    """

    x = mesh.coordinates()[:, 0]
    y = mesh.coordinates()[:, 1]

    alpha = 4*height/width**2
    sill_limit_left = center - sqrt(height/alpha)
    sill_limit_right = center + sqrt(height/alpha)

    if (sill_limit_left < 0 or
       (parameters.config['domain'] == 'square' and sill_limit_right > 1) or
       (parameters.config['domain'] == 'fjord' and sill_limit_right > parameters.config['domain_size_x'])):
        raise ValueError('Sill x-boundaries ({}, {}) lie out of domain margins ({}, {}).'
                         .format(sill_limit_left, sill_limit_right, 0, parameters.config['domain_size_x']))

    if ((parameters.config['domain'] == 'square' and height > 1) or
        (parameters.config['domain'] == 'fjord' and height > parameters.config['domain_size_y'])):
        raise ValueError('Sill y-tip {} lie out of domain margin {}.'
                         .format(height, parameters.config['domain_size_y']))

    def sill_f(x): return (-alpha * (x-center)**2) + height

    new_y = list(y)
    for i in range(len(new_y)):
        if(x[i] < sill_limit_right and x[i] > sill_limit_left):
            new_y[i] = y[i] + sill_f(x[i])*(1-y[i])
        else:
            new_y[i] = 0

    # Pointwise max to obtain not only new contour, but also correct
    # mesh outside of the sill area.
    y = np.maximum(y, new_y)

    mesh.coordinates()[:] = np.array([x, y]).transpose()

    '''def refine_mesh_at_point(self, target):
        """Refines mesh at a given point, taking points in a ball of radius mesh.hmax() around the target.

        A good resource https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1005.html"""

        self.log('Refining mesh at (%.0f, %.0f)' % (target[0], target[1]))

        to_refine = MeshFunction("bool", self.mesh, self.mesh.topology().dim() - 1)
        to_refine.set_all(False)
        mesh = self.mesh #inside `to_refine_subdomain` `self.mesh` does not exist, as `self` is redefined

        class to_refine_subdomain(SubDomain):
            def inside(self, x, on_boundary):
                return ((Point(x) - target).norm() < mesh.hmax())

        D = to_refine_subdomain()
        D.mark(to_refine, True)
        #print(to_refine.array())
        self.mesh = refine(self.mesh, to_refine)'''

    '''def refine_boundary_mesh(mesh, domain):
        """Refines mesh on ALL boundary points"""

        boundary_domain = MeshFunction("bool", mesh, mesh.topology().dim() - 1)
        boundary_domain.set_all(False)

        #Get all members of imported boundary and select only the boundary classes (i.e. exclude all other imported functions, such as from fenics).
        #Mark boundary cells as to be refined and do so.
        members = inspect.getmembers(bd, inspect.isclass) #bd is boundary module (included with this code)
        for x in members:
            if 'Bound_' in x[0]:
                obj = getattr(bd, x[0])()
                obj.mark(boundary_domain, True)

        mesh = refine(mesh, boundary_domain)

        log('Refined mesh at boundaries')

        return mesh'''
