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

    # Set up geometry + mesh.
    # FEniCS mesher seems much faster than pyGmsh for fine meshes,
    # and does not require meshio which has been problematic for Jonathan.

    # These points are used as geometry markers + to compute the ice shelf slope.
    ice_shelf_bottom_p = (0,0.05,0)
    ice_shelf_top_p = (5,1,0)
    ice_shelf_slope = (ice_shelf_top_p[1]-ice_shelf_bottom_p[1])/(ice_shelf_top_p[0]-ice_shelf_bottom_p[0])
    #print(ice_shelf_slope)

    points = [(0,0,0), (10,0,0), (10,1,0), (ice_shelf_top_p[0],1,0), ice_shelf_top_p, ice_shelf_bottom_p]
    #points = [(0,0,0), (5,0,0), (5,1,0), (0,1,0),  (0, 0.0,0)]

    ## PyGMSH mesh generation
    #g = pygmsh.built_in.Geometry()
    #pol = g.add_polygon(points, lcar=0.5)
    #mesh = pygmsh.generate_mesh(g)
    #fenics_mesh = Mesh(MPI.comm_world, mesh)

    ## FEniCS Mesher generator
    Points = [Point(p) for p in points]
    geometry = mshr.Polygon(Points)
    fenics_mesh = mshr.generate_mesh(geometry, 40)

    feffi.plot.plot_single(fenics_mesh, display=True)

    class Bound_Ice_Side(SubDomain):
        def inside(self, x, on_boundary):
            return ((0 <= x[0] <= ice_shelf_top_p[0]
                and ice_shelf_bottom_p[1] <= x[1] <= 1)
                and on_boundary)

    ## Mesh refinement ##
    refine_size = 0.2
    tolerance = 0.05 # even if using <=, >=, some points on the lines are not taken, dunno why
    class Ice_Side_Refine(SubDomain):
        def inside(self, x, on_boundary):
            return (0 <= x[0] <= ice_shelf_top_p[0]+1.5*refine_size+tolerance
                and x[1] <= ice_shelf_slope*x[0]+(ice_shelf_bottom_p[1]+tolerance)
                and x[1] >= ice_shelf_slope*x[0]+(ice_shelf_bottom_p[1]-refine_size-tolerance))

    class Cavity_Refine(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] < ice_shelf_top_p[0]+10*tolerance # totally arbitrary tolerance

    to_refine = MeshFunction("bool", fenics_mesh, fenics_mesh.topology().dim() - 1)
    to_refine.set_all(False)
    Cavity_Refine().mark(to_refine, True)
    fenics_mesh = refine(fenics_mesh, to_refine)
    feffi.flog.info('Refined mesh at cavity')
    feffi.plot.plot_single(fenics_mesh, display=True)

    to_refine = MeshFunction("bool", fenics_mesh, fenics_mesh.topology().dim() - 1)
    to_refine.set_all(False)
    Ice_Side_Refine().mark(to_refine, True)
    fenics_mesh = refine(fenics_mesh, to_refine)
    feffi.flog.info('Refined mesh at ice boundary')
    feffi.plot.plot_single(fenics_mesh, display=True)
    print(fenics_mesh.num_vertices())
    return

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

'''def generate_mesh(geom):
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
    Path(msh_name).unlink()'''

if __name__ == '__main__':
    main()
