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
import fenics


def main():
    feffi.parameters.define_parameters({
        'config_file' : 'feffi/config/simple-fjord.yml',
    })
    config = feffi.parameters.config

    ## Config shorthands
    domain_size_x = config['domain_size_x']
    domain_size_y = config['domain_size_y']
    mesh_resolution = config['mesh_resolution']
    ice_shelf_bottom_p = config['ice_shelf_bottom_p']
    ice_shelf_top_p = config['ice_shelf_top_p']
    ice_shelf_slope = config['ice_shelf_slope']

    # Set up geometry + mesh.
    # FEniCS mesher seems much faster than pyGmsh for fine meshes,
    # and does not require meshio which has been problematic for Jonathan.
    points = [(0,0,0), (domain_size_x,0,0),                            # bottom
              (domain_size_x,domain_size_y,0),                         # right
              (ice_shelf_top_p[0],domain_size_y,0), ice_shelf_top_p,   # top
              ice_shelf_bottom_p]                                      # left
    #points = [(0,0,0), (5,0,0), (5,1,0), (0,1,0),  (0, 0.0,0)]

    ## PyGMSH mesh generation
    def pygmsh2fenics_mesh(mesh):
        points = mesh.points[:,:2]

        print("Writing 2d mesh for dolfin Mesh")
        meshio.write("mesh_2d.xdmf", meshio.Mesh(
            points=points,
            cells={"triangle": mesh.cells_dict["triangle"]}))

        mesh_2d = fenics.Mesh()
        with fenics.XDMFFile("mesh_2d.xdmf") as infile:
            print("Reading 2d mesh into dolfin")
            infile.read(mesh_2d)

        return mesh_2d

    fenics_mesh = False
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(points, mesh_size=0.05)#1/mesh_resolution)

        mesh = geom.generate_mesh()
        mesh.write('mesh.xdmf')
        fenics_mesh = pygmsh2fenics_mesh(mesh)
        feffi.plot.plot_single(fenics_mesh, display=True)

    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(points, mesh_size=0.05)#1/mesh_resolution)

        # Refine
        field0 = geom.add_boundary_layer(
            edges_list=[poly.curve_loop.curves[0]],
            lcmin=0.008,
            lcmax=0.05,
            distmin=0.15, # distance up until which mesh size will be lcmin
            distmax=0.35, # distance starting at which mesh size will be lcmax
        )
        geom.set_background_mesh([field0], operator="Min")

        mesh = geom.generate_mesh()
        mesh.write('mesh.xdmf')
        fenics_mesh = pygmsh2fenics_mesh(mesh)
        #feffi.plot.plot_single(fenics_mesh, display=True)


    return


    ## FEniCS Mesher generator
    #Points = [Point(p) for p in points]
    #geometry = mshr.Polygon(Points)
    #fenics_mesh = mshr.generate_mesh(geometry, mesh_resolution)
    #fenics_mesh = UnitSquareMesh(mesh_resolution,mesh_resolution)


    class Bound_Ice_Side(SubDomain):
        def inside(self, x, on_boundary):
            return (((0 <= x[0] <= ice_shelf_top_p[0] and ice_shelf_bottom_p[1] <= x[1] <= domain_size_y)
                     or (near(x[0], ice_shelf_top_p[0]) and ice_shelf_top_p[1] <= x[1] <= domain_size_y))
                and on_boundary)


    class Ice_Side_Refine(SubDomain):
        def inside(self, x, on_boundary):
            return (0 <= x[0] <= ice_shelf_top_p[0]+1.5*refine_size+tolerance
                and x[1] <= ice_shelf_slope*x[0]+(ice_shelf_bottom_p[1]+tolerance)
                and x[1] >= ice_shelf_slope*x[0]+(ice_shelf_bottom_p[1]-refine_size-tolerance))

    class Cavity_Refine(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] < refine_size # totally arbitrary tolerance

    feffi.plot.plot_single(fenics_mesh,display=True)

    tolerance = 0.0
    refine_size = 0.2
    to_refine = MeshFunction("bool", fenics_mesh, fenics_mesh.topology().dim() - 1)
    to_refine.set_all(False)
    Ice_Side_Refine().mark(to_refine, True)
    #fenics_mesh = refine(fenics_mesh, to_refine)
    #feffi.plot.plot_single(fenics_mesh,display=True)

    refine_size = 0.04
    to_refine = MeshFunction("bool", fenics_mesh, fenics_mesh.topology().dim() - 1)
    to_refine.set_all(False)
    Ice_Side_Refine().mark(to_refine, True)
    #fenics_mesh = refine(fenics_mesh, to_refine)
    #feffi.plot.plot_single(fenics_mesh,display=True)

    # Simulation setup
    f_spaces = feffi.functions.define_function_spaces(fenics_mesh)
    f = feffi.functions.define_functions(f_spaces)
    feffi.functions.init_functions(f)

    domain = feffi.boundaries.Domain(
        fenics_mesh,
        f_spaces,
        boundaries = {
          'bottom' : feffi.boundaries.Bound_Bottom(fenics_mesh),
          'left_ice' : Bound_Ice_Side(),
          'left' : feffi.boundaries.Bound_Left(fenics_mesh),
          #'left_ice' : feffi.boundaries.Bound_Left(fenics_mesh),
          'right' : feffi.boundaries.Bound_Right(fenics_mesh),
          'top' : feffi.boundaries.Bound_Top(fenics_mesh),
        },
        BCs = feffi.parameters.config['BCs'])
    #domain.show_boundaries() # with paraview installed, will show boundary markers

    simulation = feffi.simulation.Simulation(f, domain)
    simulation.run()

    feffi.plot.plot_solutions(f, display=False)

def generate_mesh(geom):
    geo_name = 'mesh-misomip.geo'
    kwargs = {
        'prune_z_0': True,
        'remove_lower_dim_cells': True,
        'dim': 2,
        'mesh_file_type': 'msh2',
        'geo_filename': geo_name
    }
    """
    m/s
        mw: 2.2351835724163993e-08
        Tzd: 0.5955628563749149
        Szd: 11.839130634243778

    km/h
         mw: 2.5309237168319464e-08
         Tzd: 0.8116414799643376
         Szd: 9.005931824224719

        mw: 1.717010978694183e-08
      - Tzd: 2.987174408347608
      - Szd: 44.56741195772597
         """


    #if gmsh_kwargs.get('remove_faces'):
        # mutually exclusive keywords between pygmsh versions
    #    kwargs.pop('remove_lower_dim_cells')
    #    kwargs.update(gmsh_kwargs)

    mesh = pygmsh.generate_mesh(geom, **kwargs)

    from dolfin_utils.meshconvert import meshconvert

    msh_name = Path(geo_name).with_suffix('.msh')
    args = [
        "-{}".format(kwargs['dim']),
        geo_name,
        "-format",
        kwargs['mesh_file_type'],
        "-o",
        msh_name,
    ]
    gmsh_executable = kwargs.get('gmsh_path',
                                 pygmsh.helpers._get_gmsh_exe())

    p = subprocess.Popen(
        [gmsh_executable] + args,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    p.communicate()
    assert (
        p.returncode == 0
    ), "Gmsh exited with error (return code {}).".format(
        p.returncode)

    xml_name = Path(geo_name).with_suffix('.xml')
    meshconvert.convert2xml(msh_name, xml_name)
    mesh = str(xml_name.resolve())

    print("Removing ", geo_name)
    Path(geo_name).unlink()
    print("Removing ", msh_name)
    Path(msh_name).unlink()

    return mesh


if __name__ == '__main__':
    main()

