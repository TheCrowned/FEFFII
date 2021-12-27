# feffi module lives one level up current dir
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import feffi
from fenics import *
import mshr
import matplotlib.pyplot as plt
import numpy as np
import shutil
import subprocess
import pygmsh
from fenics import Mesh, XDMFFile, MPI, plot
import meshio
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile


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

    ##############################
    ### Set up geometry + mesh ###
    ##############################

    # FEniCS mesher seems much faster than pyGmsh for fine meshes, but
    # pyGmsh allows fine grained control over mesh size at different locations.

    # pygmsh handles long boundary refinement badly, so we need to interval the
    # ice shelf boundary with many points, see
    # https://github.com/nschloe/pygmsh/issues/506
    shelf_points_x = list(np.arange(ice_shelf_bottom_p[0], ice_shelf_top_p[0], 0.1))
    shelf_points_x.reverse()
    shelf_points = [(x, ice_shelf_f(x), 0) for x in shelf_points_x[0:-1]] # exclude last (bottom) point to avoid duplicate

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
        poly = geom.add_polygon(points, mesh_size=1/mesh_resolution)

        # Refine
        refine_lines = [poly.curve_loop.curves[i] for i in range(len(points)-len(shelf_points)-3, len(points)-1)]
        field0 = geom.add_boundary_layer(
            edges_list=refine_lines,
            lcmin=0.03,
            lcmax=0.5,
            distmin=0.10, # distance up until which mesh size will be lcmin
            distmax=0.20, # distance starting at which mesh size will be lcmax
        )
        field1 = geom.add_boundary_layer(
            edges_list=refine_lines,
            lcmin=0.005,
            lcmax=0.5,
            distmin=0.01, # distance up until which mesh size will be lcmin
            distmax=0.10, # distance starting at which mesh size will be lcmax
        )
        geom.set_background_mesh([field0, field1], operator="Min")

        mesh = geom.generate_mesh()
        mesh.write('mesh.xdmf')
        fenics_mesh = pygmsh2fenics_mesh(mesh)
        #print(fenics_mesh.num_vertices())

    #feffi.plot.plot_single(fenics_mesh, display=True)

    class Bound_Ice_Shelf(SubDomain):
        def inside(self, x, on_boundary):
            return (((0 <= x[0] <= ice_shelf_top_p[0] and ice_shelf_bottom_p[1] <= x[1] <= domain_size_y)
                     or (near(x[0], ice_shelf_top_p[0]) and ice_shelf_top_p[1] <= x[1] <= domain_size_y))
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
          'ice_shelf' : Bound_Ice_Shelf(),
          'left' : feffi.boundaries.Bound_Left(fenics_mesh),
          'right' : feffi.boundaries.Bound_Right(fenics_mesh),
          'top' : feffi.boundaries.Bound_Top(fenics_mesh),
        },
        BCs = feffi.parameters.config['BCs'])
    #domain.show_boundaries() # with paraview installed, will show boundary markers

    simulation = feffi.simulation.Simulation(f, domain)
    simulation.run()

    # Use this instead of simulation.run() if you wanna see 3eqs system output
    #for i in range(10):
    #    simulation.timestep()
        #feffi.boundaries.visualize_f_on_boundary(simulation.mw, domain, 'left')
        #feffi.boundaries.visualize_f_on_boundary(simulation.Tzd, domain, 'left')
        #feffi.boundaries.visualize_f_on_boundary(simulation.Szd, domain, 'left')

    feffi.plot.plot_solutions(f, display=False)


def pygmsh2fenics_mesh(mesh):
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
