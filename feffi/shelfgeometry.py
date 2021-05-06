"""
.. module:: shelfgeometry
   :synopsis: This module contains the `class::ShelfGeometry` that
   generates semi-structured meshes for the typical shelf-ocean
   geometry that Stefano Ottolenghi uses in his Boussinesq simulations
   of oceans in contact with an ice shelf.

.. moduleauthor:: Christian Helanow <christian.helanow@math.su.se>

"""
import pygmsh
import meshio
import tempfile
import shutil
import logging
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from fenics import Mesh, XDMFFile, MPI
# Python >= 3.4
from pathlib import Path


class ShelfGeometry(object):
    """Creates a mesh for a simple shelf geometry using pyGMSH.

    Creates a quasi-rectangular domain of size (0, `domain_parameters`[0])x(0,
    `domain_parameters`[1]), with a 'ice shelf', i.e. the absence of a part
    of the mesh, of width `domain_parameters`[2] and thickness
    `domain_parameters`[3] located at the upper left corner.

    In general, to describe the number of layers in a given part of the domain
    (`ny_shelf`,`ny_ocean`, `nx`), either an int or a list of
    coordinates can be provided. If an int, the (specified) part of
    the domain will be subdivided into int layers. If a list, this
    must contain x or y coordinates where the layers should be
    located.

    For the horizontal meshing (base):
    If `lc_list` is None, then mesh layers will located at the
    specified coordinate points (in addition to the layers that bound
    the domain) specified by `nx`.
    If `lc_list` is given it must be a list of characteristic mesh
    sizes (see characteristic lc in GMSH), and the length of the list
    must equal the length of `nx`.
    If `nx` is None, `lc_list` specifies the characteristic mesh size
    of the points located at x = left side of the domain, width of the
    ice shelf, right side of the domain (i.e., 0,
    `domain_parameters`[2], `domain_parameters`[0])
    If `ny_shelf` or `domain_parameters[3]` equal 0, a rectangular
    mesh will be returned.

    Parameters
    ----------
    domain_parameters : list or tuple
        Must contain 4 entries that describe the general layout of the
        domain in the following order: total width of domain, total
        height of domain, width/length of ice shelf, thickness of ice shelf.
    ny_ocean : int or list
        Describes the number of vertical subdivisions of the mesh from
        the base surface up to the base of the ice shelf (i.e. the
        number of vertical subdivision in the part of the ocean that
        underlies the 'ice shelf') . If an int, this will be
        `ny_ocean` number of layers in the vertical. If a list, this
        must contain the coordinates at which a layer is to be located.
    ny_shelf : int or list
        Describes the number of vertical subdivisions of the mesh from
        the top surface down to the base of the ice shelf (i.e. the
        number of vertical subdivision in the part of the ocean that
        is parallel to the 'ice shelf') . If an int, this will be
        `ny_shelf` number of layers in the vertical. If a list, this
        must contain the coordinates at which a layer is to be
        located. If 0 or None, the domain will be rectangular without
        a shelf part with `ny_ocean` layers in the vertical.
    nx : int or list, optional
    log_level : int, optional
        Verbosity level, either 0, 1 (default) or 2.

    Attributes
    ----------
    geom : pygmsh Geometry
        Geometry created from input parameters that is to be used to
        create the mesh.
    mesh : pygmsh generated mesh
        Default None. If mesh has been generated pygmsh.Mesh.

    Raises
    ------
    ValueError :
        If both `nx` and `lc_list` are None, of different lengths etc.
    """
    def __init__(self, domain_params,
                 ny_ocean, ny_shelf,
                 nx=None, lc_list=None,
                 log_level=1):
        super(ShelfGeometry, self).__init__()
        self.domain_width = domain_params[0]
        self.domain_height = domain_params[1]
        self.shelf_width = domain_params[2]
        self.shelf_thickness = domain_params[3]
        self.ny_shelf = ny_shelf
        self.ny_ocean = ny_ocean
        self.geom = pygmsh.built_in.Geometry()
        self.mesh = None
        self.log_level = log_level

        # basic checks
        if not (nx or lc_list):
            raise ValueError("Please specify at least `nx` or `lc_list`")
        _arg_err = False
        if not nx or isinstance(nx, int):
            self.nx = nx
        elif isinstance(nx, (tuple, list, np.ndarray)):
            self.nx = list(nx)
        elif not nx:
            pass
        else:
            _arg_err = True
        if isinstance(lc_list, (tuple, list, np.ndarray)):
            self.lc_list = list(lc_list)
        elif not lc_list:
            self.lc_list = lc_list
        else:
            _arg_err = True
        if _arg_err:
            raise ValueError(
                "`nx` must be int or array like and `lc_list` array like")
        # for making (basic) rectangle
        if not self.ny_shelf or self.shelf_thickness == 0:
            self.ny_shelf = 0
            self.shelf_thickness = 0

        if self.nx and not self.lc_list:
            # extruded base or nx base
            if isinstance(self.nx, int):
                ratio = self.shelf_width/self.domain_width
                nx_shelf = round(ratio * self.nx)
                nx_ocean = self.nx - nx_shelf
                print(
                    "Dividing the base mesh into",
                    "{} cells over the shelf".format(nx_shelf),
                    "and {} cells over the open ocean.".format(nx_ocean))
                shelf_points = list(np.linspace(
                    0, self.shelf_width, nx_shelf + 1))
                ocean_points = list(np.linspace(
                    self.shelf_width, self.domain_width,
                    nx_ocean + 1))
                # skip the double mid points
                base_points = shelf_points + ocean_points[1:]
                # self.extruded_base_geometry(shelf_points, ocean_points)
            else:
                if not self.shelf_width in nx:
                    raise ValueError(
                        "The x-coordinate of the shelf front must be in the set `nx`")
                self.nx.sort()
                base_points = self.nx
                if base_points[0] != 0:
                    base_points = [0] + base_points
                if base_points[-1] != self.domain_width:
                    base_points += [self.domain_width]
                # shelf_points = self.nx[:self.nx.index(self.shelf_width) + 1]
                # ocean_points = self.nx[self.nx.index(self.shelf_width):]
                # self.extruded_base_geometry(shelf_points, ocean_points)
            lc_points = [1e10,] * len(base_points)
            self.lc_base_geometry(base_points, lc_points)
        elif self.nx and self.lc_list:
            if not (len(self.nx) == len(self.lc_list)
                and self.nx[0] == 0
                and self.nx[-1] == self.domain_width
                and self.shelf_width in nx):
                raise ValueError(
                    "`nx` and `lc_list` must be of same length > 3 and"
                    "`nx` must contain the x-coordinates of the domain "
                    "start, end and shelf front")
            self.lc_list = [self.lc_list[self.nx.index(i)] for i in sorted(self.nx)]
            self.nx = list(self.nx)
            self.nx.sort()
            # lc base
            self.lc_base_geometry(self.nx, self.lc_list)
        elif self.lc_list:
            if len(self.lc_list) != 3:
                raise ValueError(
                    "`lc_list` must specify characteristic length of existing point")
            base_points = [0, self.shelf_width, self.domain_width]
            self.lc_base_geometry(base_points, self.lc_list)

    def plot_mesh(self):
        """Plots the pygmsh-generated mesh."""

        if not self.mesh:
            print("No mesh to plot...")
            return
        points = self.mesh.points
        triangles = self.mesh.get_cells_type('triangle')
        if not triangles.size:
            raise ValueError("Can only plot 2d meshes")
        pts = points[:, :2]
        for e in triangles:
            for idx in [[0, 1], [1, 2], [2, 0]]:
                X = pts[e[idx]]
                plt.plot(X[:, 0], X[:, 1], "-k")
        plt.gca().set_aspect("equal", "datalim")
        plt.axis("off")
        plt.show()

    def _normalize_layer_list(self, layers, min_coor, max_coor):
        """Normalizes the list of coordinates to layers in [0, 1] for
        gmsh layer extrusion.

        Parameters
        ----------
        layers : list
                List of physical coordinates in *one* direction, i.e.
                list of floats.
        min_coor/max_coor : float
                The minimum/maximum physical coordinate value of the
                domain in the direction the
                mesh is to be extruded.

        Parameters
        ----------
        """
        assert (
            isinstance(layers, list)
        ), "Layers to be normalized must be a list."
        layers.sort()
        if not (layers[0] >= min_coor and layers[-1] <= max_coor):
            raise ValueError(
                "Specified extrusion layers incompatible with domain args.")
        last_layer = [] if layers[-1] == max_coor else [1.0]
        layers = [(x - min_coor)/(max_coor - min_coor)
                  for x in layers] + last_layer
        return layers

    def _get_num_layers(self, num_layers, min_coor, max_coor,
                        subdivisions=None):
        """Gets the number of layers in a suitable format for gmsh extrusion.

        Parameters
        ----------
        layers : list
                List of physical coordinates in *one* direction, i.e.
                list of floats.
        min_coor/max_coor : float
                The minimum/maximum physical coordinate value of the
                domain in the direction the
                mesh is to be extruded.
        subdivisions : list of ints, optional
                Default None. If specified, uses the number of
                subdivision for each specified layer.
        """
        if isinstance(num_layers, int):
            return num_layers
        num_layers = self._normalize_layer_list(
            num_layers, min_coor, max_coor)
        if not subdivisions:
            layers = ("{{{}}}".format(",".join(['1']*len(num_layers)))
                      + ", {{{}}}".format(",".join(map(str, num_layers))))
        else:
            assert (
                len(num_layers) == len(subdivisions)
            ), "The number of layers and subdivisions must be of equal length"
            layers = ("{{{}}}".format(",".join(map(str, subdivisions)))
                      + ", {{{}}}".format(",".join(map(str, num_layers))))
        return layers

    def extruded_base_geometry(self, shelf_points, ocean_points):
        """Extrudes the geometry in the x-direction (base mesh) along
        the given points.

        shelf_points/ocean_points : list of floats
                List of floats representing coordinates at which there
                should be a division layer in the shelf/ocean domain
                part of the horizontal mesh.
        """
        assert (shelf_points[-1] == self.shelf_width and
                ocean_points[0] == self.shelf_width
                ), "x-coordinated paritioning and geometry mismatch"
        p0 = self.geom.add_point([0, 0, 0], 1e20)
        p1, line1, __ = self.geom.extrude(
            p0, translation_axis=[self.shelf_width, 0, 0],
            num_layers=self._get_num_layers(
                shelf_points, 0, self.shelf_width))
        p2, line2, __ = self.geom.extrude(
            p1, translation_axis=[self.domain_width-self.shelf_width, 0, 0],
            num_layers=self._get_num_layers(
                ocean_points, self.shelf_width, self.domain_width))
        line3, _, _ = self.geom.extrude(
            line1,
            translation_axis=[0, self.domain_height-self.shelf_thickness, 0],
            num_layers=self._get_num_layers(
                self.ny_ocean, 0, self.domain_height-self.shelf_thickness))
        line4, _, _ = self.geom.extrude(
            line2,
            translation_axis=[0, self.domain_height-self.shelf_thickness, 0],
            num_layers=self._get_num_layers(
                self.ny_ocean, 0, self.domain_height-self.shelf_thickness))
        # extrude ocean part beyond shelf
        if self.ny_shelf:
            line5, _, _ = self.geom.extrude(
                line4, translation_axis=[0, self.shelf_thickness, 0],
                num_layers=self._get_num_layers(
                    self.ny_shelf, self.domain_height-self.shelf_thickness,
                    self.domain_height))

    def lc_base_geometry(self, base_points, lc_points):
        """Using characteristic length for the horizontal base mesh.

        Parameters
        ----------
        base_points : list
                List of coordinate points in the x-direction wher
                characteristic mesh size (in the x-dirction) should be
                specified. *Must* contain the point at the left side,
                right side and shelf_width of the domain.
        lc_points : list
                List of characteristic lenghts at the specified
                `base_points`. If a 'large enough' values is chosen
                for this, meshing will only occur at the points.
        """
        shelf_index = base_points.index(self.shelf_width)
        base_lines = []
        points = [self.geom.add_point([p, 0, 0], lc)
                  for p, lc in zip(base_points, lc_points)]
        p0 = points[0]
        for p1 in points[1:]:
            l = self.geom.add_line(p0, p1)
            base_lines.append(l)
            p0 = p1
        top_lines = []
        for l in base_lines:
            top_line, _, _ = self.geom.extrude(
                l, translation_axis=[
                    0, self.domain_height-self.shelf_thickness, 0],
                num_layers=self._get_num_layers(
                    self.ny_ocean, 0, self.domain_height-self.shelf_thickness))
            top_lines.append(top_line)
        # extrude ocean part beyond shelf
        if self.ny_shelf:
            for l in top_lines[shelf_index:]:
                _, _, _ = self.geom.extrude(
                    l, translation_axis=[0, self.shelf_thickness, 0],
                    num_layers=self._get_num_layers(
                        self.ny_shelf, self.domain_height-self.shelf_thickness,
                        self.domain_height))

    def generate_mesh(self, use_dolfin_xml=False,
                      **gmsh_kwargs):
        """Generates the pygmsh mesh from the constructed geometry.

        .. note ::
                It seems that meshio can write extruded meshes in a
                way so that 'internal boundaries' remain, or the mesh
                topology is screwed up somehow. Fenics does not find
                all the boundary facets and can mark internal facets.
                As a temporary fix, use dolfin-convert.

        Parameters
        ----------
        use_dolfin_xml: bool, optional
                Whether to use the dolfing legacy format for mesh file
                (xml). Default is False.

        Kwargs
        ------
        gmsh_kwargs : dict
                Keyword arguments to be used with `pygmsh.generate_mesh`
        """
        if use_dolfin_xml:
            self.is_mesh_xml = True
        else:
            self.is_mesh_xml = False
        with tempfile.NamedTemporaryFile(suffix='.geo') as f:
            geo_name = f.name

        kwargs = {
            'prune_z_0': True,
            'remove_lower_dim_cells': True,
            # 'extra_gmsh_arguments': ["-format", "msh2"],
            'dim': 2,
            'mesh_file_type': 'msh2',
            'geo_filename': geo_name
        }

        kwargs['verbose'] = True if self.log_level > 1 else False

        if gmsh_kwargs.get('remove_faces'):
            # mutually exclusive keywords between pygmsh versions
            kwargs.pop('remove_lower_dim_cells')
        kwargs.update(gmsh_kwargs)
        try:
            mesh = pygmsh.generate_mesh(self.geom,
                                        **kwargs)
        except TypeError as e:
            # switch argument for old pygmsh version < 6.1.1 (can't
            # get version number)
            if e.__str__().rfind('remove_lower_dim_cells') == -1:
                raise
            else:
                kwargs['remove_faces'] = kwargs.pop('remove_lower_dim_cells')
                self.mesh = pygmsh.generate_mesh(self.geom, dim=2,
                                                 **kwargs)
        except UnboundLocalError as e:
            e.msg += "\nThere seems to be no geometry to mesh."
            raise
        finally:
            print("Removing ", geo_name)
            Path(geo_name).unlink()
        if use_dolfin_xml:
            # import here since this is a poor hack
            try:
                from dolfin_utils.meshconvert import meshconvert
            except ImportError as e:
                e.msg += "\ndolfin XML format cannot be used without meshconvert."
                raise
            msh_name = Path(geo_name).with_suffix('.msh')
            args = [
                "-{}".format(kwargs['dim']),
                geo_name,
                "-format",
                kwargs['mesh_file_type'],
                "-o",
                msh_name,
            ] + kwargs['extra_gmsh_arguments']
            gmsh_executable = kwargs.get('gmsh_path',
                                         pygmsh.helpers._get_gmsh_exe())
            try:
                p = subprocess.Popen(
                    [gmsh_executable] + args,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                )
            except FileNotFoundError:
                print("Is gmsh installed?")
                raise
            p.communicate()
            assert (
                p.returncode == 0
            ), "Gmsh exited with error (return code {}).".format(
                p.returncode)

            logging.warning("Dolfin XML is a legacy format. "
                            "This is only to be used due to a bug in meshio somewhere.")
            xml_name = Path(geo_name).with_suffix('.xml')
            try:
                meshconvert.convert2xml(msh_name, xml_name)
                mesh = str(xml_name.resolve())
            except Exception as e:
                raise
            finally:
                print("Removing ", geo_name)
                Path(geo_name).unlink()
                print("Removing ", msh_name)
                Path(msh_name).unlink()
        self.mesh = mesh

    def get_fenics_mesh(self, save_name=None):
        """Retrieves the fenics mesh of the pygmsh generated mesh.

        Parameters
        ----------
        save_name : string, optional
                If None (default), a temporary xdmf file will be
                created from the `mesh` attribute. Otherwise should be
                a path to which specifying where the xdmf file should
                be saved.
        """
        save_suffix = '.xml' if self.is_mesh_xml else '.xdmf'
        if not self.mesh:
            raise ValueError("No mesh has been created.")
        if not save_name and self.is_mesh_xml:
            file_name = self.mesh
        elif not save_name and not self.is_mesh_xml:
            with tempfile.NamedTemporaryFile(suffix=save_suffix) as f:
                file_name = f.name
        else:
            suffix = Path(save_name).suffix
            if not suffix:
                if self.is_mesh_xml:
                    file_name = save_name + '.xml'
                    shutil(self.mesh, file_name)
                    Path(self.mesh).unlink()
                else:
                    file_name = save_name + '.xdmf'
            elif ((suffix == '.xdmf' and not self.is_mesh_xml) or
                  (suffix == '.xml' and self.is_mesh_xml)):
                file_name = save_name
            else:
                raise ValueError("Save file must be an '.xdmf' file.")
        if self.is_mesh_xml:
            fenics_mesh = Mesh(MPI.comm_world, file_name)
        else:
            meshio.write(file_name, self.mesh)
            with XDMFFile(MPI.comm_world,
                          file_name) as xdmf_infile:
                fenics_mesh = Mesh(MPI.comm_world)
                xdmf_infile.read(fenics_mesh)
        if not save_name:
            Path(file_name).unlink()
        return fenics_mesh
