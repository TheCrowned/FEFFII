import shutil
import subprocess
import pygmsh
from fenics import Mesh, XDMFFile, MPI, plot
import meshio
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile

pts = [(799500.0, 79500.0), (640500.0, 79500.0), (639500.0, 67500.0), (636500.0, 67500.0), (635500.0, 63500.0), (608500.0, 63500.0), (607500.0, 59500.0), (588500.0, 59500.0), (587500.0, 55500.0), (587500.0, 24500.0), (588500.0, 20500.0), (607500.0, 20500.0), (608500.0, 16500.0), (635500.0, 16500.0), (636500.0, 12500.0), (639500.0, 12500.0), (640500.0, 500.0), (799500.0, 500.0)]
pts = [(i[0], i[1], 0) for i in pts]

def main():
    g = pygmsh.built_in.Geometry()
    pol = g.add_polygon(pts)
    g.extrude(pol, [0,0,1], num_layers=3)
    mesh = generate_mesh(g)
    #mesh = pygmsh.generate_mesh(g, geo_filename='mesh-misomip.geo')
    #fenics_mesh = get_fenics_mesh(mesh)
    fenics_mesh = Mesh(MPI.comm_world, 'mesh-misomip.xml')
    plot(fenics_mesh)
    plt.show()

def generate_mesh(geom):
    geo_name = 'mesh-misomip.geo'
    kwargs = {
        'prune_z_0': True,
        'remove_lower_dim_cells': True,
        'dim': 3,
        'mesh_file_type': 'msh2',
        'geo_filename': geo_name
    }

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

def get_fenics_mesh(mesh, save_name=None):
    """Retrieves the fenics mesh of the pygmsh generated mesh.

    Parameters
    ----------
    save_name : string, optional
            If None (default), a temporary xdmf file will be
            created from the `mesh` attribute. Otherwise should be
            a path to which specifying where the xdmf file should
            be saved.
    """
    save_suffix = '.xml'
    with tempfile.NamedTemporaryFile(suffix=save_suffix) as f:
        file_name = f.name


        fenics_mesh = Mesh(MPI.comm_world, file_name)

        Path(file_name).unlink()
    return fenics_mesh

if __name__ == '__main__':
    main()