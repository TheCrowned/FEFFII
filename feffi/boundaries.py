from fenics import MeshFunction, File, DirichletBC, Constant, Expression, SubDomain, near
from os import system
from numpy import max
import logging
from . import parameters
flog = logging.getLogger('feffi')

class Domain(object):
    """ Creates a simulation domain given its boundaries definitions and
        corresponding boundary conditions.

        Calls `self.define_boundaries`, `self.mark_boundaries` and
        `self.define_BCs` in sequence.
        Boundaries are marked before BCs are set on them.

        Parameters
        ----------
        mesh : a fenics compatible mesh object
        f_spaces : dict
                Function spaces dictionary, for example as output by
                `feffi.functions.define_function_spaces()`.

        kwargs
        ------
        boundaries : dict
                A custom geometry can be used by providing a dictionary of
                the form `{label : SubDomainMethod}`. Labels should match the
                ones provided in the BCs dict, and the SubDomainMethod(s) should
                be defined somewhere, if not among the native ones.
        BCs : dict
                Dictionary defining boundary conditions. Should contain one
                sub-dictionary per each function space, with matching labels.
                Each sub-dictionary should have keys matching `boundaries`
                keys and values either int/floats, or strings if they are
                to be compiled into a `fenics.Expression`.
                If you want to set a point-wise BC, enter the relevant point
                in the form of a tuple as BC label.
                Note: velocity BCs should be vector valued and provided as
                a list. If you want NO BC to be enforced on one component,
                write `'null'`.

        Examples
        --------
        1)  Square with in/outflow on right side:

            mesh = feffi.mesh.create_mesh(domain='square')
            f_spaces = feffi.functions.define_function_spaces()
            domain = feffi.boundaries.Domain(
                mesh,
                f_spaces,
                BCs = {
                  'top' : [0, 'null']
                  'right' : [0, '0.5*sin(2*pi*x[1])']
                  'bottom' : [0, 0]
                  'left' : [0, 0]
                },
                domain='square')

            # domain.BCs contains the BCs ready to be enforced in simulation

        2)  Custom geometry, with different subdomains+BCs on right side:

            class Bound_Bottom_Right(fenics.SubDomain):
                def inside(self, x, on_boundary):
                    return near(x[0], 1) and x[1] < 0.5 and on_boundary
            class Bound_Top_Right(fenics.SubDomain):
                def inside(self, x, on_boundary):
                    return near(x[0], 1) and x[1] >= 0.5 and on_boundary

            domain = feffi.boundaries.Domain(
                mesh,
                f_spaces,
                boundaries = {
                  'top' : feffi.boundaries.Bound_Top(),
                  'top_right' : Bound_Top_Right(),
                  'bottom_right' : Bound_Bottom_Right(),
                  'bottom' : feffi.boundaries.Bound_Bottom(),
                  'left' : feffi.boundaries.Bound_Left()
                },
                BCs = {
                  'V' : {
                    'top' : [0, 'null'],
                    'top_right' : [0, '2*x[1]-1'],
                    'bottom_right' : [0, '-2*x[1]+1'],
                    'bottom' : [0, 0],
                    'left' : [0, 0]
                  },
                  'Q' : {
                    '(0,0)' : 0
                  }
                })
        """

    def __init__(self, mesh, f_spaces, boundaries = {}, **kwargs):
        self.mesh = mesh
        self.f_spaces = f_spaces
        self.boundaries = boundaries

        # Allow function arguments to overwrite wide config (but keep it local)
        self.config = dict(parameters.config); self.config.update(kwargs)

        # Check BCs status
        if(not isinstance(self.config['BCs'], dict)):
            flog.error('No BCs given (at least not as dict).')
        else:
            if(not self.config['BCs'].get('V') or
               not self.config['BCs'].get('Q') or
               not self.config['BCs'].get('T') or
               not self.config['BCs'].get('S')):
                flog.warning(
                    'BCs were only given for {} spaces.'.format(
                        ', '.join(list(self.config['BCs'].keys()))))

        # If no custom domain is provided, assume one of default cases
        if self.boundaries == {}:
            self.define_boundaries()

        self.mark_boundaries()
        self.define_BCs()

    def define_boundaries(self):
        """Defines boundaries as SubDomains.
        """

        self.boundaries = {
            'right' : Bound_Right(),
            'bottom' : Bound_Bottom(),
            'left' : Bound_Left()
        }

        # Define subdomains
        if self.config['domain'] == 'square':
            self.boundaries.update({
                'top' : Bound_Top()
            })
        elif self.config['domain'] == 'fjord':
            self.boundaries.update({
                'ice_shelf_bottom' : Bound_Ice_Shelf_Bottom(),
                'ice_shelf_right' : Bound_Ice_Shelf_Right(),
                'sea_top' : Bound_Sea_Top()
            })

    def mark_boundaries(self):
        """
        Boundaries are marked through a `fenics.MeshFunction`, which is
        useful for BCs setting in later mesh deformation.
        Association between boundaries and markers is stored into
        `self.subdomains_markers`.
        """
        # Mark subdomains and store this matching
        self.marked_subdomains = MeshFunction(
            "size_t",
            self.mesh,
            self.mesh.topology().dim() - 1)

        self.subdomains_markers = {}
        i = 1
        for (name, subdomain) in self.boundaries.items():
            print('subdomain {} name {} marked with marker {}'.format(subdomain, name, i))
            subdomain.mark(self.marked_subdomains, i)
            self.subdomains_markers[name] = i
            i += 1

    def show_boundaries(self):
        """
        Exports boundaries data and open Paraview for display.

        Unfortunately there doesn't seem to be a way to just `plot` them, see:
        https://fenicsproject.discourse.group/t/
        plotting-the-boundary-function-of-a-2d-mesh/140
        """

        File("boundaries.pvd") << self.marked_subdomains
        system('paraview "boundaries.pvd"')
        # It would be nice to call
        # `paraview --data=boundaries.pvd --script=SMTH`
        # with a python script that would apply the view, maybe.

    def define_BCs(self):
        """Defines boundary conditions."""

        self.BCs = {}

        for (f_space_name, BCs_set) in self.config['BCs'].items():
            self.BCs[f_space_name] = []

            for (subdomain_name, BC_value) in BCs_set.items():

                # Vector valued function spaces (i.e. velocity) should have
                # BCs applied on both components, if given.
                if self.f_spaces[f_space_name].num_sub_spaces() != 0:
                    for i in range(self.f_spaces[f_space_name].num_sub_spaces()):
                        if BC_value[i] != 'null':
                            self.BCs[f_space_name].append(
                                DirichletBC(
                                    self.f_spaces[f_space_name].sub(i),
                                    self.parse_BC(BC_value[i]),
                                    self.marked_subdomains,
                                    self.subdomains_markers[subdomain_name]
                                ))
                            flog.info(
                                ('BCs - Boundary {}, space {}[{}] ' +
                                 '(marker {}), value {}').format(
                                    subdomain_name, f_space_name, i,
                                    self.subdomains_markers[subdomain_name],
                                    BC_value[i]))

                # Scalar valued function spaces BCs
                else:

                    # If label correspond to a subdomain, apply to that
                    if self.subdomains_markers.get(subdomain_name):
                        self.BCs[f_space_name].append(
                            DirichletBC(
                                self.f_spaces[f_space_name],
                                self.parse_BC(BC_value),
                                self.marked_subdomains,
                                self.subdomains_markers[subdomain_name]
                            ))
                        flog.info(
                        ('BCs - Boundary {}, space {} ' +
                         '(marker {}), value {}').format(
                            subdomain_name, f_space_name,
                            self.subdomains_markers[subdomain_name],
                            BC_value))

                    # Otherwise, we assume it's a pointwise condition,
                    # and evaluate the label expecting a tuple back.
                    # It makes sense to apply pointwise BCS only to pressure,
                    # but we leave it general.
                    else:
                        point = eval(subdomain_name)
                        self.BCs[f_space_name].append(
                            DirichletBC(
                                self.f_spaces[f_space_name],
                                self.parse_BC(BC_value),
                                'near(x[0], {}) && near(x[1], {})'.format(
                                    point[0], point[1]),
                                method='pointwise'
                            ))
                        flog.info(
                            'BCs - Point ({}, {}), space Q, value {}'.format(
                                point[0], point[1], BC_value))


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


### SUBDOMAIN DEFINITIONS ###

class Bound_Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1) and on_boundary

class Bound_Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0) and on_boundary

class Bound_Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0) and on_boundary

class Bound_Right(SubDomain):
    def inside(self, x, on_boundary):
        if parameters.config['domain'] == 'square':
            return near(x[0], 1) and on_boundary
        elif parameters.config['domain'] == 'fjord':
            return near(x[0], parameters.config['domain_size_x']) \
            and on_boundary

class Bound_Ice_Shelf_Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return  x[0] >= 0 and x[0] <= parameters.config['shelf_size_x'] and \
                near(x[1], parameters.config['domain_size_y'] - parameters.config['shelf_size_y']) \
                and on_boundary

class Bound_Ice_Shelf_Right(SubDomain):
    def inside(self, x, on_boundary):
        return  near(x[0], parameters.config['shelf_size_x']) and \
                x[1] >= parameters.config['domain_size_y'] - parameters.config['shelf_size_y'] and \
                x[1] <= parameters.config['domain_size_y'] and on_boundary

class Bound_Sea_Top(SubDomain):
    def inside(self, x, on_boundary):
        return  x[0] >= parameters.config['shelf_size_x'] and \
                near(x[1], parameters.config['domain_size_y']) and on_boundary