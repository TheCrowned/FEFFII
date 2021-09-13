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
        'config_file' : 'feffi/config/john-setup.yml',
    })

    #points = [(0,0,0), (1,0,0), (1,1,0), (0,1,0)]#, (0, 0.1,0)]
    points = [(0,0,0), (5,0,0), (5,1,0), (1,1,0), (0, 0.1,0)]
    g = pygmsh.built_in.Geometry()
    pol = g.add_polygon(points, lcar=0.5)
    mesh = generate_mesh(g)
    fenics_mesh = Mesh(MPI.comm_world, 'mesh-misomip.xml')
    #plot(fenics_mesh)
    #plt.show()

    f_spaces = feffi.functions.define_function_spaces(fenics_mesh)
    f = feffi.functions.define_functions(f_spaces)
    feffi.functions.init_functions(f)

    class Bound_Left(SubDomain):
        def inside(self, x, on_boundary):
            return (((0 <= x[0] <= 1 and near(x[1], 0.9*x[0]+0.1))
                #or (10 <= x[0] <= 20 and near(x[1], 0.01*(x[0]-10)+1))
                or (near(x[0], 0) and 0 <= x[1] <= 0.1))
                and on_boundary)

    domain = feffi.boundaries.Domain(
        fenics_mesh,
        f_spaces,
        boundaries = {
          'bottom' : feffi.boundaries.Bound_Bottom(fenics_mesh),
          'left' : Bound_Left(),
          #'left' : feffi.boundaries.Bound_Left(fenics_mesh),
          'right' : feffi.boundaries.Bound_Right(fenics_mesh),
          'top' : feffi.boundaries.Bound_Top(fenics_mesh),
        },
        BCs = {
          'V' : {
            'bottom' : [0, 0],
            'left' : [0, 0],
            'right' : [0,0],
            'top' : [0,0]
          },
          'Q' : {
            '(1,1)' : 0
          },
          'T' : {},
          'S' : {}
        })
    domain.show_boundaries()

    simulation = feffi.simulation.Simulation(f, domain)
    #for i in range(200):
    #    simulation.timestep()
    #feffi.plot.plot_solutions(f, display=True)
    simulation.run()
    feffi.plot.plot_solutions(f)

def generate_mesh(geom):
    geo_name = 'mesh-misomip.geo'
    kwargs = {
        'prune_z_0': True,
        'remove_lower_dim_cells': True,
        'dim': 2,
        'mesh_file_type': 'msh2',
        'geo_filename': geo_name
    }
    '''
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
         '''


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

if __name__ == '__main__':
    main()
