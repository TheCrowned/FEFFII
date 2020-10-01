import fenics, logging
from shelfgeometry import ShelfGeometry
import feffi.parameters

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
    config = feffi.parameters.config
    config_l = dict(config); config_l.update(kwargs)

    if config_l['domain'] == "square":
        mesh = fenics.UnitSquareMesh(config_l['mesh_resolution'], config_l['mesh_resolution'])

    if config_l['domain'] == "custom":
        # general domain geometry: width, height, ice shelf width, ice shelf thickness
        domain_params = [config_l['domain_size_x'], config_l['domain_size_y'], config_l['shelf_size_x'], config_l['shelf_size_y']]

        sg = ShelfGeometry(
            domain_params,
            ny_ocean = config_l['mesh_resolution_y'],          # layers on "deep ocean" (y-dir)
            ny_shelf = config_l['mesh_resolution_sea_y'],      # layers on "ice-shelf thickness" (y-dir)
            nx = config_l['mesh_resolution_x'],              # layers x-dir
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

