# feffi module lives one level up current dir
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import feffi
from fenics import *
import numpy as np
import pygmsh
from fenics import Mesh, XDMFFile, MPI, plot
import meshio
import matplotlib.pyplot as plt

def main():
    feffi.parameters.define_parameters({
        'config_file' : 'feffi/config/mitgcm-setup.yml',
    })
    config = feffi.parameters.config

    ## Config shorthands
    domain_size_x = config['domain_size_x']
    domain_size_y = config['domain_size_y']
    mesh_resolution = config['mesh_resolution']
    ice_shelf_bottom_p = config['ice_shelf_bottom_p']
    ice_shelf_top_p = config['ice_shelf_top_p']
    ice_shelf_slope = config['ice_shelf_slope']
    ice_shelf_f = lambda x: ice_shelf_slope*x+ice_shelf_bottom_p[1]
    lcar = mesh_resolution
    step_x = 0.1
    step_y = 0.05

    ##############################
    ### Set up geometry + mesh ###
    ##############################

    # FEniCS mesher seems much faster than pyGmsh for fine meshes, but
    # pyGmsh allows fine grained control over mesh size at different locations.

    # pygmsh handles long boundary refinement badly, so we need to interval the
    # ice shelf boundary with many points, see
    # https://github.com/nschloe/pygmsh/issues/506
    # It is not clear whether the step size has an influence on the mesh quality
    shelf_points_x = list(np.arange(ice_shelf_bottom_p[0], ice_shelf_top_p[0], 0.4))
    shelf_points_x.reverse()
    shelf_points = [(x, ice_shelf_f(x), 0) for x in shelf_points_x[0:-1]] # exclude last (bottom) point to avoid duplicate
    #print(shelf_points)
    
    points = [(0,0,0), (domain_size_x,0,0),                           # bottom
              (domain_size_x,domain_size_y,0),                        # right
              (ice_shelf_top_p[0],domain_size_y,0), ice_shelf_top_p]  # sea top
    points += shelf_points                                            # ice shelf
    points += [ice_shelf_bottom_p]                                    # left

    ##############################
    ### PyGMSH mesh generation ###
    ##############################

    fenics_mesh = False
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(points, mesh_size=mesh_resolution)
        '''origin = geom.add_point([0, 0, 0], lcar)
        br = geom.add_point([domain_size_x, 0, 0], lcar)
        tr = geom.add_point([domain_size_x, 1, 0], lcar)
        ice_top = geom.add_point([ice_shelf_top_p[0], domain_size_y, 0], lcar)
        ice_top2 = geom.add_point([ice_shelf_top_p[0], ice_shelf_top_p[1], 0], lcar)
        ice_bottom = geom.add_point([ice_shelf_bottom_p[0]+2*step_x, ice_shelf_bottom_p[1]+step_y, 0], lcar)
        ice_bottom2 = geom.add_point([ice_shelf_bottom_p[0]+step_x, ice_shelf_bottom_p[1]+step_y/5, 0], lcar)
        ice_bottom3 = geom.add_point([ice_shelf_bottom_p[0], ice_shelf_bottom_p[1], 0], lcar)

        l1 = geom.add_line(origin, br)
        l2 = geom.add_line(br, tr)
        l3 = geom.add_line(tr, ice_top)
        l4 = geom.add_line(ice_top, ice_top2)
        l5 = geom.add_line(ice_top2, ice_bottom)
        l6 = geom.add_spline([ice_bottom, ice_bottom2, ice_bottom3])
        l7 = geom.add_line(ice_bottom3, origin)

        ll = geom.add_curve_loop([l1,l2,l3,l4, l5, l6, l7])
        pl = geom. add_plane_surface(ll)'''

        # Mesh refinement
        ice_shelf_lines = [poly.curve_loop.curves[i] for i in range(len(points)-len(shelf_points)-3, len(points)-1)]
        #left_lines = [poly.curve_loop.curves[i] for i in range(len(points)-1, len(points))]

        # Progressive mesh refinement
        '''field0 = geom.add_boundary_layer(
            edges_list=ice_shelf_lines,
            lcmin=0.02,
            lcmax=0.04,
            distmin=0.08, # distance up until which mesh size will be lcmin
            distmax=0.15, # distance starting at which mesh size will be lcmax
        )
        field1 = geom.add_boundary_layer(
            edges_list=ice_shelf_lines,
            lcmin=0.005,
            lcmax=0.02,
            distmin=0.01, # distance up until which mesh size will be lcmin
            distmax=0.08, # distance starting at which mesh size will be lcmax
        )
        field2 = geom.add_boundary_layer(
            edges_list=left_lines,
            lcmin=0.005,
            lcmax=0.5,
            distmin=0.01, # distance up until which mesh size will be lcmin
            distmax=0.10, # distance starting at which mesh size will be lcmax
        )

        geom.set_background_mesh([field0, field1, field2], operator="Min")
        '''

        # One single refinement, bit less controlled but maybe less error-prone?
        field = geom.add_boundary_layer(
            edges_list=ice_shelf_lines,
            lcmin=0.006,
            lcmax=mesh_resolution, #0.15,
            distmin=0.009, # distance up until which mesh size will be lcmin
            distmax=0.3, # distance starting at which mesh size will be lcmax
        )
        geom.set_background_mesh([field], operator="Min")

        # Generate mesh
        mesh = geom.generate_mesh()
        mesh.write('mesh.xdmf')
        fenics_mesh = pygmsh2fenics_mesh(mesh)
        #fenics_mesh = UnitSquareMesh(50,50)
        print(fenics_mesh.num_vertices())

    feffi.plot.plot_single(fenics_mesh, display=True)

    class Bound_Ice_Shelf_Top(SubDomain):
        def inside(self, x, on_boundary):
            return (((0.0 <= x[0] <= ice_shelf_top_p[0] and ice_shelf_bottom_p[1] <= x[1] <= domain_size_y)
                     or (near(x[0], ice_shelf_top_p[0]) and ice_shelf_top_p[1] <= x[1] <= domain_size_y))
                and on_boundary)
    class Bound_Ice_Shelf_Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return (((0 <= x[0] <= 0.05 and ice_shelf_bottom_p[1] <= x[1] <= domain_size_y))
                    and on_boundary)

    # Simulation setup
    f_spaces = feffi.functions.define_function_spaces(fenics_mesh)
    f = feffi.functions.define_functions(f_spaces)
    feffi.functions.init_functions(f)

    domain = feffi.boundaries.Domain(
        fenics_mesh,
        f_spaces,
        boundaries = {
          'bottom' : feffi.boundaries.Bound_Bottom(fenics_mesh),
          'ice_shelf_bottom' : Bound_Ice_Shelf_Bottom(),
          'ice_shelf_top' : Bound_Ice_Shelf_Top(),
          'left' : feffi.boundaries.Bound_Left(fenics_mesh),
          'right' : feffi.boundaries.Bound_Right(fenics_mesh),
          'top' : feffi.boundaries.Bound_Top(fenics_mesh),
        },
        BCs = feffi.parameters.config['BCs'])
    #domain.show_boundaries() # with paraview installed, will show boundary markers

    simulation = feffi.simulation.Simulation(f, domain)
    simulation.run()

    feffi.plot.plot_solutions(f, display=False)


def pygmsh2fenics_mesh(mesh):
    """Convert mesh from PyGMSH output to FEniCS."""
    
    points = mesh.points[:,:2] # must strip z-coordinate for fenics

    print("Writing 2d mesh for dolfin Mesh")
    meshio.write("mesh_2d.xdmf", meshio.Mesh(
        points=points,
        cells={"triangle": mesh.cells_dict["triangle"]}))

    fenics_mesh_2d = Mesh()
    with XDMFFile("mesh_2d.xdmf") as infile:
        print("Reading 2d mesh into dolfin")
        infile.read(fenics_mesh_2d)

    return fenics_mesh_2d


if __name__ == '__main__':
    main()
