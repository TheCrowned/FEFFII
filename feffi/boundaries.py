from fenics import (MeshFunction, File, DirichletBC, Constant, Expression,
                    SubDomain, near, BoundaryMesh, ALE)
from os import system
import numpy as np
import logging
import itertools
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

    def __init__(self, mesh, f_spaces, boundaries={}, **kwargs):
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
        self.store_subdomains_indexes()
        self.define_BCs()

    def define_boundaries(self):
        """Defines boundaries as SubDomains."""

        self.boundaries = {
            'right': Bound_Right(),
            'left': Bound_Left(),
            'bottom': Bound_Bottom(self.mesh.geometric_dimension()),
        }

        # For 3D, add the 2 extra domains
        if self.mesh.geometric_dimension() == 3:
            self.boundaries.update({
                'front': Bound_Front(),
                'back': Bound_Back(),
            })

        # Define subdomains
        if self.config['domain'] == 'square':
            self.boundaries.update({
                'top': Bound_Top(self.mesh.geometric_dimension()),
            })
        elif self.config['domain'] == 'fjord':
            self.boundaries.update({
                'ice_shelf_bottom': Bound_Ice_Shelf_Bottom(self.mesh.geometric_dimension()),
                'ice_shelf_right': Bound_Ice_Shelf_Right(self.mesh.geometric_dimension()),
                'sea_top': Bound_Sea_Top(self.mesh.geometric_dimension()),
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
            subdomain.mark(self.marked_subdomains, i)
            self.subdomains_markers[name] = i
            i += 1

    def store_subdomains_indexes(self):
        """
        Stores a dictionary containing the association between each subdomain
        and the mesh points that belong to it, with the respective indexes.
        """

        self.subdomains_points = {}
        b_mesh = BoundaryMesh(self.mesh, 'exterior')

        for (name, subdomain) in self.boundaries.items():
            b_points = {}

            for p_idx in range(len(b_mesh.coordinates())):
                if subdomain.inside(b_mesh.coordinates()[p_idx], True):
                    b_points[p_idx] = b_mesh.coordinates()[p_idx]

            b_orientation = get_boundary_orientation(list(b_points.values()))
            print('boundary {} oriented wrt to {} coord'.format(name, b_orientation))
            b_points = dict(sorted(b_points.items(),
                                   key=lambda dict_entry: dict_entry[1][b_orientation]))
            self.subdomains_points[name] = b_points

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

    def deform_boundary(self, boundary_label, goal_profile):
        """

        """

        def closest_k_nodes(node, nodes, k):
            result = {}
            nodes = np.asarray(nodes)
            mask = np.ones(len(nodes), dtype=bool)
            distances = np.sum((nodes-node)**2, axis=1)

            closest_node_idx = -1
            max_dist = -1
            for _ in range(k):
                prev_closest_node_idx = closest_node_idx
                closest_node_idx = np.argmin(distances[mask])

                if prev_closest_node_idx > -1 and closest_node_idx >= prev_closest_node_idx:
                    closest_node_idx += 1

                result[closest_node_idx] = {'coord': nodes[closest_node_idx], 'dist': distances[closest_node_idx]}
                mask[closest_node_idx] = False

            return result

        # Deform boundary mesh and move whole mesh according to new boundary
        b_mesh = BoundaryMesh(self.mesh, 'exterior')
        for (p_idx, p_coord) in self.subdomains_points[boundary_label].items():
            #p_coord = b_mesh.coordinates()[p_idx]
            closest = closest_k_nodes(p_coord, goal_profile, self.mesh.geometric_dimension())

            denom = 0
            avg = 0
            for entry in closest.values():
                if entry['dist'] == 0:
                    continue

                denom += 1/entry['dist']
                avg += (1/entry['dist'])*entry['coord']

            new_p = avg/denom

            print(('old point {} becomes {}'
                  .format(b_mesh.coordinates()[p_idx], new_p)))

            b_mesh.coordinates()[p_idx][0] = new_p[0]
            b_mesh.coordinates()[p_idx][1] = new_p[1]
            if self.mesh.geometric_dimension() == 3:
                b_mesh.coordinates()[p_idx][2] = new_p[2]

        ALE.move(self.mesh, b_mesh)
        self.mesh.bounding_box_tree().build(self.mesh)


        '''goal_profile_length = curve_length(goal_profile)

        # Current left profile
        curr_profile = self.subdomains_points[boundary_label]
        curr_profile_length = curve_length(list(curr_profile.values()))

        print(('Current mesh left profile has {} vertexes, perimeter {} and coordinates \n {}\n\n'
               .format(len(curr_profile), curr_profile_length, curr_profile)))
        print(('Goal mesh left profile has {} vertexes, perimeter {} and coordinates \n {}'
               .format(len(goal_profile), goal_profile_length, goal_profile)))

        # Obtain the new ocean profile
        goal_p_idx = 0
        curr_profile_iter = itertools.cycle(list(curr_profile.items()))
        curr_profile_curr_p = next(curr_profile_iter)
        new_profile = {curr_profile_curr_p[0]: goal_profile[0]} # first point matches both for ice and ocean
        new_profile_last_p = goal_profile[0]

        # Goal mesh is supposed to be finer than current mesh.
        # First task: to obtain a goal mesh of equal grane as current one.
        for i in range(1, len(list(curr_profile.items()))-1):
            curr_profile_prev_p_idx = curr_profile_curr_p[0]
            curr_profile_prev_p_coord = curr_profile_curr_p[1]

            curr_profile_curr_p = next(curr_profile_iter)
            curr_profile_curr_p_idx = curr_profile_curr_p[0]
            curr_profile_curr_p_coord = curr_profile_curr_p[1]

            d = np.linalg.norm(np.subtract(curr_profile_curr_p_coord, curr_profile_prev_p_coord))/curr_profile_length
            direction_vector = np.subtract(goal_profile[goal_p_idx+1], new_profile_last_p)
            new_p = new_profile_last_p + np.multiply(direction_vector, d*goal_profile_length/np.linalg.norm(direction_vector))
            new_profile[curr_profile_curr_p_idx] = new_p

            new_profile_last_p = new_p
            if new_profile_last_p[1] >= goal_profile[goal_p_idx+1][1]: # first coord forces it to be vertical boundary
                goal_p_idx += 1

        # set last point, also matching ice exactly
        curr_profile_curr_p = next(curr_profile_iter)
        new_profile[curr_profile_curr_p[0]] = goal_profile[-1]

        # new_profile contains now the new ocean profile coming from the ice. We'll take gradual steps to reach it

        total_steps = 1 # n of steps during which we want to spread the deformation out
        for x in range(total_steps):
            deform_coeff = (x+1)/total_steps #how much to deform the mesh in this single t step

            #simul.timestep() #run one ocean timestep

            # Deform boundary mesh and move whole mesh according to new boundary
            b_mesh = BoundaryMesh(self.mesh, 'exterior')
            for (p_idx, p_coord) in new_profile.items():
                print(('old point {} becomes {}'
                      .format(b_mesh.coordinates()[p_idx],
                              (deform_coeff*new_profile[p_idx][0],
                               new_profile[p_idx][1]))))

                b_mesh.coordinates()[p_idx][0] = deform_coeff*new_profile[p_idx][0]
                b_mesh.coordinates()[p_idx][1] = new_profile[p_idx][1]

            ALE.move(self.mesh, b_mesh)
            self.mesh.bounding_box_tree().build(self.mesh)'''

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

                        if self.mesh.geometric_dimension() == 2:
                            if len(point) != 2:
                                raise ValueError('Expecting 2D BC for pressure.')

                            application_point = ('near(x[0], {}) && near(x[1], {})'
                                                 .format(point[0], point[1]))
                            flog.info('BCs - Point ({}, {}), space Q, value {}'
                                      .format(point[0], point[1], BC_value))

                        elif self.mesh.geometric_dimension() == 3:
                            if len(point) != 3:
                                raise ValueError('Expecting 3D BC for pressure.')

                            application_point = ('near(x[0], {}) && near(x[1], {}) && near(x[2], {})'
                                                 .format(point[0], point[1], point[2]))
                            flog.info('BCs - Point ({}, {}, {}), space Q, value {}'
                                      .format(point[0], point[1], point[2], BC_value))

                        self.BCs[f_space_name].append(
                            DirichletBC(
                                self.f_spaces[f_space_name],
                                self.parse_BC(BC_value),
                                application_point,
                                method='pointwise'
                        ))

    def parse_BC(self, BC):
        """Parses a single string-represented BC into a Fenics-ready one.

        Makes a given BC into either a fenics.Constant or a fenics.Expression
        of degree 2.
        """

        try:
            parsed_BC = Constant(float(BC))
        except ValueError:
            parsed_BC = Expression(BC, degree=2)

        return parsed_BC


### SUBDOMAIN DEFINITIONS ###

# When initializing (some) subdomains, pass the geometrical dimension of the domain.
# The meaning of `x[1]` changes depending on whether it's 2D or 3D.

class Bound_Top(SubDomain):
    def __init__(self, dim):
        self.dim = dim
        super().__init__()

    def inside(self, x, on_boundary):
        if self.dim == 2:
            return near(x[1], 1) and on_boundary
        elif self.dim == 3:
            return near(x[2], 1) and on_boundary


class Bound_Bottom(SubDomain):
    def __init__(self, dim):
        self.dim = dim
        super().__init__()

    def inside(self, x, on_boundary):
        if self.dim == 2:
            return near(x[1], 0) and on_boundary
        elif self.dim == 3:
            return near(x[2], 0) and on_boundary


class Bound_Front(SubDomain): # 3D only
    def inside(self, x, on_boundary):
        return near(x[1], 0) and on_boundary


class Bound_Back(SubDomain): # 3D only
    def inside(self, x, on_boundary):
        return near(x[1], 1) and on_boundary


class Bound_Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0) and on_boundary


class Bound_Right(SubDomain):
    def inside(self, x, on_boundary):
        if parameters.config['domain'] == 'square':
            return near(x[0], 1) and on_boundary
        elif parameters.config['domain'] == 'fjord':
            return (near(x[0], parameters.config['domain_size_x'])
                    and on_boundary)


class Bound_Ice_Shelf_Bottom(SubDomain):
    def __init__(self, dim):
        self.dim = dim
        super().__init__()

    def inside(self, x, on_boundary):
        if self.dim == 2:
            return (x[0] >= 0 and x[0] <= parameters.config['shelf_size_x'] and
                    near(x[1], parameters.config['domain_size_y'] - parameters.config['shelf_size_y'])
                    and on_boundary)
        elif self.dim == 3:
            return (x[0] >= 0 and x[0] <= parameters.config['shelf_size_x'] and
                    near(x[2], parameters.config['domain_size_y'] - parameters.config['shelf_size_y'])
                    and on_boundary)


class Bound_Ice_Shelf_Right(SubDomain):
    def __init__(self, dim):
        self.dim = dim
        super().__init__()

    def inside(self, x, on_boundary):
        if self.dim == 2:
            return (near(x[0], parameters.config['shelf_size_x']) and
                    x[1] >= parameters.config['domain_size_y'] - parameters.config['shelf_size_y'] and
                    x[1] <= parameters.config['domain_size_y'] and on_boundary)
        elif self.dim == 3:
            return (near(x[0], parameters.config['shelf_size_x']) and
                    x[2] >= parameters.config['domain_size_y'] - parameters.config['shelf_size_y'] and
                    x[2] <= parameters.config['domain_size_y'] and on_boundary)


class Bound_Sea_Top(SubDomain):
    def __init__(self, dim):
        self.dim = dim
        super().__init__()

    def inside(self, x, on_boundary):
        if self.dim == 2:
            return (x[0] >= parameters.config['shelf_size_x'] and
                    near(x[1], parameters.config['domain_size_y']) and on_boundary)
        elif self.dim == 3:
            return (x[0] >= parameters.config['shelf_size_x'] and
                    near(x[2], parameters.config['domain_size_y']) and on_boundary)

def get_boundary_orientation(points):
    """
    Determine boundary orientation (i.e. whether it is vertical/horizontal).
    This does not generalize well to 3D, as the problem is ambigouos there.

    Parameters
    ----------
    points : list
         Points belonging to the boundary.

    Output
    ------
    d : int
        Coordinate of growth.

    Examples
    --------
    l = [(0,0), (0,0.2), (0,0.4), (0,0.6), (0,0.8), (0,1)]
    get_boundary_orientation(l) # will return 1
    """

    points = np.array(points)
    dim = len(points[0])

    for d in range(dim):
        # Assume we are only interested in x- and z-oriented boundaries.
        # In 3D, we accept to get garbage result for y-oriented boundaries.
        if d == 1 and dim == 3:
            continue

        if min(points[:,d]) != max(points[:,d]): #this doesn't generalize to 3d
            return d

    raise ValueError('Given points list does not seem to have an orientation.')

def curve_length(curve_points):
    '''Naive curve length calculation with pythagorean theorem.'''

    profile_length = 0
    for i in range(1, len(curve_points)):
        profile_length += np.linalg.norm(np.subtract(curve_points[i],
                                                     curve_points[i-1]))
    return profile_length