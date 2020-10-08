from fenics import MeshFunction, File, DirichletBC, Constant, Expression
from os import system
from numpy import max
import logging
from . import parameters
from boundaries import square as domain_square
from boundaries import fjord as domain_fjord

class Domain(object):
    """ Creates a simulation domain given its boundaries definitions and
        corresponding boundary conditions.

        Calls `self.define_boundaries` and `self.define_BCs` in sequence.
        Boundaries are marked before BCs are set on them.

        Parameters
        ----------
        mesh : a fenics compatible mesh object
        f_spaces : dict
                Function spaces dictionary, for example as output by
                `feffi.functions.define_function_spaces()`.

        kwargs
        ------
        BCs : dict
                Dictionary defining boundary conditions. Should contain one
                sub-dictionary per each function space, with matching labels.
                Each sub-dictionary should have keys matching `boundaries`
                keys and values either int/floats, or strings if they are
                to be compiled into a `fenics.Expression`.
                Note: velocity BCs should be vector valued and provided as
                a list. If you want NO BC to be enforced on one component,
                write `'null'`.
                Note: pressure BCs are not supported. A default BC of null
                pressure is automatically applied on the top-right corner.

        Examples
        --------
        1)  Square with in/outflow on right side:

            mesh = feffi.mesh.create_mesh(domain='square')
            f_spaces = feffi.functions.define_function_spaces()
            domain = feffi.boundaries.Domain(
                mesh,
                f_spaces,
                { 'top' : [0, 'null']
                  'right' : [0, '0.5*sin(2*pi*x[1])']
                  'bottom' : [0, 0]
                  'left' : [0, 0]
                },
                domain='square'
            )

            # domain.BCs contains the BCs ready to be enforced in simulation
        """

    def __init__(self, mesh, f_spaces, **kwargs):
        self.mesh = mesh
        self.f_spaces = f_spaces

        # Allow function arguments to overwrite wide config (but keep it local)
        self.config = dict(parameters.config); self.config.update(kwargs)

        self.define_boundaries()
        self.define_BCs()

    def define_boundaries(self):
        """Defines boundaries as SubDomains.

        Boundaries are marked through a `fenics.MeshFunction`, which is useful
        for BCs setting in later mesh deformation.
        Association between boundaries and markers is stored into `self.subdomains_markers`.
        """

        # Define subdomains
        if(self.config['domain'] == 'square'):
            subdomains = {
                'right' : domain_square.Bound_Right(),
                'bottom' : domain_square.Bound_Bottom(),
                'left' : domain_square.Bound_Left(),
                'top' : domain_square.Bound_Top()
            }
        elif(self.config['domain'] == 'fjord'):
            subdomains = {
                'right' : domain_fjord.Bound_Right(),
                'bottom' : domain_fjord.Bound_Bottom(),
                'left' : domain_fjord.Bound_Left(),
                'ice_shelf_bottom' : domain_fjord.Bound_Ice_Shelf_Bottom(),
                'ice_shelf_right' : domain_fjord.Bound_Ice_Shelf_Right(),
                'sea_top' : domain_fjord.Bound_Sea_Top()
            }

        # Mark subdomains and store this matching
        i = 1
        self.marked_subdomains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.subdomains_markers = {}
        for (name, subdomain) in subdomains.items():
            subdomain.mark(self.marked_subdomains, i)
            self.subdomains_markers[name] = i
            i += 1

    def show_boundaries(self):
        """
        Exports boundaries data and open Paraview for display.

        Unfortunately there doesn't seem to be a way to just `plot` them, see:
        https://fenicsproject.discourse.group/t/plotting-the-boundary-function-of-a-2d-mesh/140
        """

        File("boundaries.pvd") << self.marked_subdomains
        system('paraview "boundaries.pvd"')
        # It would be nice to call `paraview --data=boundaries.pvd --script=SMTH` with a python script that would apply the view, maybe.

    def define_BCs(self):
        """Defines boundary conditions.
        """

        self.BCs = {}

        for (f_space_name, BCs_set) in self.config['BCs'].items():
            self.BCs[f_space_name] = []

            for (subdomain_name, BC_value) in BCs_set.items():
                if self.f_spaces[f_space_name].num_sub_spaces() != 0:
                    for i in range(self.f_spaces[f_space_name].num_sub_spaces()):
                        if BC_value[i] != 'null':
                            self.BCs[f_space_name].append(
                                DirichletBC(
                                    self.f_spaces[f_space_name].sub(i),
                                    self.parse_BC(BC_value[i]),
                                    self.marked_subdomains,
                                    self.subdomains_markers[subdomain_name]
                                )
                            )
                            logging.info('BCs - Boundary %s, space %s[%d] (marker %d), value %s' % (subdomain_name, f_space_name, i, self.subdomains_markers[subdomain_name], BC_value[i]))
                else:
                    self.BCs[f_space_name].append(
                        DirichletBC(
                            self.f_spaces[f_space_name],
                            self.parse_BC(BC_value),
                            self.marked_subdomains,
                            self.subdomains_markers[subdomain_name]
                        )
                    )
                    logging.info('BCs - Boundary %s, space %s (marker %d), value %s' % (subdomain_name, f_space_name, self.subdomains_markers[subdomain_name], BC_value))

        top_right_corner = max(self.mesh.coordinates(), 0)
        """Works because of how mesh vertices are ordered, for example:
        >>> m.coordinates()
        array([[0. , 0. ],
               [0.3, 0. ],
               [0.6, 0. ],
               ...
               [0.0, 0.1],
               [0.3, 0.1],
               ...
               [0.0, 0.9],
               ...
               [2.7, 1. ],
               [3. , 1. ]])
        """

        self.BCs['Q'] = [
            DirichletBC(
                self.f_spaces['Q'],
                Constant(0),
                'near(x[0], %f) && near(x[1], %f)' % (top_right_corner[0], top_right_corner[1]), #np.max returns max over given axis in the form of a n-1 dimensional array, from which we only need
                method='pointwise'
            )
        ]
        logging.info('BCs - Top-right corner (%f, %f), space Q, value 0' % (top_right_corner[0], top_right_corner[1]))

    def parse_BC(self, BC):
        """Parses a single string-represented BC into a Fenics-ready one.

        Makes a given BC into either a fenics.Constant or a fenics.Expression
        of degree 2.
        """

        try:
            parsed_BC = Constant(float(BC))
        except ValueError:
            parsed_BC = Expression(BC, degree = 2)

        return parsed_BC