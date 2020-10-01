from fenics import MeshFunction, File, DirichletBC, Constant, Expression
from os import system
from numpy import max

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
        boundaries : dict
                Dictionary defining boundaries as `fenics.SubDomain`s or as
                to-be-`fenics.CompiledSubDomain`. If boundaries are defined
                through sub-classes of `fenics.SubDomain`, they should all
                live inside an external module (manually) imported as
                `user_imported_boundaries`.
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

            import examples.boundaries.square as user_imported_boundaries
            mesh = fenics.UnitSquareMesh(1,1)
            f_spaces = feffi.functions.define_function_spaces()
            domain = feffi.boundaries.Domain(
                mesh,
                f_spaces,
                { 'top' : 'Square_Bound_Top'
                  'right' : 'Square_Bound_Right'
                  'bottom' : 'Square_Bound_Bottom'
                  'left' : 'near(x[0], 1)' #example of mixed formulation
                },
                { 'top' : [0, 'null']
                  'right' : [0, '0.5*sin(2*pi*x[1])']
                  'bottom' : [0, 0]
                  'left' : [0, 0]
                },
            )

            #domain.BCs contains the BCs ready to be enforced in simulation
        """

    def __init__(self, mesh, f_spaces, boundaries, BCs):
        self.mesh = mesh

        self.define_boundaries(boundaries)
        self.define_BCs(f_spaces, BCs)

    def define_boundaries(self, boundaries):
        """Defines boundaries as SubDomains.

        Boundaries are marked through a `fenics.MeshFunction`, which is useful
        for BCs setting in later mesh deformation.
        Association between boundaries and markers ints is stored into `self.subdomains_markers`.

        Parameters
        ----------
        boundaries : dict; see `__init__`.
        """

        #Associate a marker integer to each boundary, and store this matching
        i = 1
        self.subdomains_markers = dict(boundaries)
        for (name, val) in self.subdomains_markers.items():
            self.subdomains_markers[name] = i
            i += 1

        #Effectively mark subdomains
        self.marked_subdomains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        for (name, subdomain_callback) in boundaries.items():
            subdomain = getattr(user_imported_boundaries, subdomain_callback)()
            # ! TODO check if subdomain_callback is really an existent function, or if we should compile it as a CompiledSubDomain
            subdomain.mark(self.marked_subdomains, self.subdomains_markers[name])

    def show_boundaries(self):
        """
        Exports boundaries data and open Paraview for display.

        Unfortunately there doesn't seem to be a way to just `plot` them, see:
        https://fenicsproject.discourse.group/t/plotting-the-boundary-function-of-a-2d-mesh/140
        """

        File("boundaries.pvd") << self.marked_subdomains
        system('paraview "boundaries.pvd"')
        # It would be nice to call `paraview --data=boundaries.pvd --script=SMTH` with a python script that would apply the view, maybe.

    def define_BCs(self, f_spaces, BCs):
        """Define boundary conditions.

        Parameters
        ----------
        f_spaces : dict
        BCs : dict; see `__init__`.
        """

        self.BCs = {}

        for (f_space_name, BCs_set) in BCs.items():
            for (subdomain_name, BC_value) in BCs_set.items():
                self.BCs[f_space_name].append(
                    DirichletBC(
                        f_spaces[f_space_name],
                        self.parse_BC(BC_value),
                        self.marked_subdomains,
                        self.subdomains_markers[subdomain_name]
                    )
                )
                # ! TODO support velocity BC on one component only

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
                f_spaces['Q'],
                Constant(0),
                'near(x[0], %f) && near(x[1], %f)' % (top_right_corner[0], top_right_corner[1]), #np.max returns max over given axis in the form of a n-1 dimensional array, from which we only need
                method='pointwise'
            )
        ]
