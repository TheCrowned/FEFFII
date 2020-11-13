import fenics, logging
from feffi.shelfgeometry import ShelfGeometry
from . import parameters

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
        mesh = fenics.UnitSquareMesh(config['mesh_resolution'], config['mesh_resolution'])

    if config['domain'] == "fjord":
        # general domain geometry: width, height, ice shelf width, ice shelf thickness
        domain_params = [config['domain_size_x'], config['domain_size_y'], config['shelf_size_x'], config['shelf_size_y']]

        sg = ShelfGeometry(
            domain_params,
            ny_ocean = config['mesh_resolution_y'],          # layers on "deep ocean" (y-dir)
            ny_shelf = config['mesh_resolution_sea_y'],      # layers on "ice-shelf thickness" (y-dir)
            nx = config['mesh_resolution_x'],              # layers x-dir
        )

        sg.generate_mesh()
        mesh = sg.get_fenics_mesh()

        #self.refine_mesh_at_point(Point(self.args.shelf_size_x, self.args.domain_size_y - self.args.shelf_size_y))

        '''fenics_domain = Rectangle(Point(0., 0.), Point(1., 1.)) - \
                        Rectangle(Point(0.0, 0.9), Point(0.4, 1.0))
        mesh = generate_mesh(fenics_domain, resolution, "cgal")
        deform_mesh_coords(mesh)

        mesh = refine_mesh_at_point(mesh, Point(0.4, 0.9), domain)'''

    logging.info('Initialized mesh: vertexes %d, max diameter %.2f' % (mesh.num_vertices(), mesh.hmax()))

    return mesh

    '''def mesh_add_sill(self, center, height, length):
        """Deforms mesh coordinates to create the bottom bump"""

        x = self.mesh.coordinates()[:, 0]
        y = self.mesh.coordinates()[:, 1]

        alpha = 4*height/length**2
        sill_function = lambda x : ((-alpha*(x - center)**2) + height)
        sill_left = center - sqrt(height/alpha)
        sill_right = center + sqrt(height/alpha)

        new_y = [y[i] + sill_function(x[i])*(1-y[i]) if(x[i] < sill_right and x[i] > sill_left) else 0 for i in range(len(y))]
        y = np.maximum(y, new_y)

        self.mesh.coordinates()[:] = np.array([x, y]).transpose()

        #self.sill = {'f':sill_function, 'left':sill_left, 'right':sill_right}
        #self.bd.sill = self.sill

    def refine_mesh_at_point(self, target):
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
