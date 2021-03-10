from fenics import (dot, inner, elem_mult, grad, nabla_grad, div,
                    dx, ds, sym, Identity, Function, TrialFunction,
                    TestFunction, FunctionSpace, VectorElement, split,
                    FiniteElement, Constant, interpolate, Expression)
import fenics
from . import parameters
import logging
flog = logging.getLogger('feffi')
f = {}

def define_function_spaces(mesh, **kwargs):
    """Define function spaces for velocity, pressure, temperature and salinity.

    Parameters
    ----------
    mesh : a fenics-compatible mesh object
        Mesh on which to define function spaces.

    Return
    ------
    function_spaces : dictionary
    """

    # Allow function arguments to overwrite wide config (but keep it local)
    config = dict(parameters.config); config.update(kwargs)

    V = VectorElement("Lagrange", mesh.ufl_cell(), parameters.config['degree_V'])
    P = FiniteElement("Lagrange", mesh.ufl_cell(), parameters.config['degree_P'])

    f_spaces = {
        'W': FunctionSpace(mesh, V * P),
        'T': FunctionSpace(mesh, 'CG', parameters.config['degree_T']),
        'S': FunctionSpace(mesh, 'CG', parameters.config['degree_S'])
    }

    # Store V and P separately for convenience (mostly for BCs setting)
    f_spaces['V'] = f_spaces['W'].sub(0)
    f_spaces['Q'] = f_spaces['W'].sub(1)

    return f_spaces

def define_functions(f_spaces):
    """Define solution functions for velocity, pressure, temperature
    and salinity.

    Parameters
    ----------
    f_spaces : dict
        Function spaces for velocity, pressure, temperature and salinity.

    Return
    ------
    functions : dictionary
    """

    global f

    # Define functions needed for solution computation
    f = {
        #'u_n': Function(f_spaces['V']),
        #'u_': Function(f_spaces['V']),
        #'u': TrialFunction(f_spaces['V']),
        #'v': TestFunction(f_spaces['V']),
        #'p_n': Function(f_spaces['Q']),
        #'p_': Function(f_spaces['Q']),
        #'p': TrialFunction(f_spaces['Q']),
        #'q': TestFunction(f_spaces['Q']),
        'T_n': Function(f_spaces['T']),
        'T_': Function(f_spaces['T']),
        'T': TrialFunction(f_spaces['T']),
        'T_v': TestFunction(f_spaces['T']),
        'S_n': Function(f_spaces['S']),
        'S_': Function(f_spaces['S']),
        'S': TrialFunction(f_spaces['S']),
        'S_v': TestFunction(f_spaces['S'])
    }

    (f['u'], f['p']) = split(TrialFunction(f_spaces['W']))
    (f['u_'], f['p_']) = split(Function(f_spaces['W']))
    (f['u_n'], f['p_n']) = split(Function(f_spaces['W']))
    (f['v'], f['q']) = split(TestFunction(f_spaces['W']))
    f['sol'] = Function(f_spaces['W'])

    # Nice names for output (ex. in Paraview)
    #f['u_'].rename("velocity", "Velocity in m/s")
    #f['p_'].rename("pressure", "Pressure in Pa")
    #f['T_'].rename("temperature", "Temperature in °C")
    #f['S_'].rename("salinity", "Salinity in PSU")

    return f

def init_functions(f, **kwargs):
    """Set function values to closest stable state to speed up convergence.

    Parameters
    ----------
    f : dict
        Functions to initialize
    kwargs : `T_0`, `S_0`, `rho_0`, `g` (refer to README for info).
    """

    # Allow function arguments to overwrite wide config (but keep it local)
    config = dict(parameters.config); config.update(kwargs)

    '''f['u_n'].assign(
        fenics.interpolate(
            fenics.Expression(
                (0, '(x[0])*0.5*sin(2*pi*x[1])'),
                degree=2
            ),
            f['u_n'].ufl_function_space()))
    f['T_n'].assign(
        fenics.interpolate(
            fenics.Expression(
                '(1-x[0])*1',
                degree=2,
                T_0=config['T_0']
            ),
            f['T_n'].ufl_function_space()))'''
    '''f['T_n'].assign(
        interpolate(
            Constant(config['T_0']),
            f['T_n'].ufl_function_space()))
    f['S_n'].assign(
        interpolate(
            Constant(config['S_0']),
            f['S_n'].ufl_function_space()))
    f['p_n'].assign(
        interpolate(
            Expression(
                'rho_0*g*(1-x[1])',
                degree=2,
                rho_0=config['rho_0'],
                g=config['g']),
            f['p_n'].ufl_function_space()))'''

def N(a, u, p):
    """First stabilization operator, LHS differential operator.
    Corresponds to operator L(U) in LBB paper."""

    step_size = 1/parameters.config['steps_n']
    nu = parameters.config['nu'][0] #parameters.assemble_viscosity_tensor(parameters.config['nu'])[0]
    rho_0 = parameters.config['rho_0']

    return u/step_size - nu*div(nabla_grad(u)) + dot(a, nabla_grad(u)) + grad(p)/rho_0

def Phi(a, u):
    """Second stabilization operator.
    Corresponds to operator Phi in LBB paper."""

    return dot(a, nabla_grad(u))

def B_g(a, u, p, v, q):
    """Galerkin weak formulation for Navier-Stokes."""

    step_size = 1/parameters.config['steps_n']
    nu = parameters.config['nu'][0]# parameters.assemble_viscosity_tensor(parameters.config['nu'])
    rho_0 = parameters.config['rho_0']
    n = fenics.FacetNormal(a.function_space().mesh())

    return (
    + (1/step_size)*dot(u, v)*dx
    + nu*inner(nabla_grad(u), nabla_grad(v))*dx # *nu!!
    + (dot(dot(a, nabla_grad(u)), v) )*dx
    - dot(p/rho_0, div(v))*dx
    - dot(p/rho_0, dot(v, n))*ds
    - nu*dot(dot(nabla_grad(u), n), v)*ds
    - dot(div(u), q)*dx )

def build_buoyancy(T_, S_):
    """Build buoyancy term."""

    return Expression((0, '-g*(1 - beta*(T_ - T_0) + gamma*(S_ - S_0))'), # g is given positive
        beta = parameters.config['beta'], gamma = parameters.config['gamma'],
        T_0 = parameters.config['T_0'], S_0 = parameters.config['S_0'],
        g = parameters.config['g'],
        T_ = T_, S_ = S_,
        degree=2)

def build_NS_GLS_steady_form(a, u, u_n, p, v, q, delta, tau, T_, S_):
    """Build Navier-Stokes steady state weak form + GLS stabilization."""

    step_size = 1/parameters.config['steps_n']

    b = build_buoyancy(T_, S_)
    f = u_n/step_size + b
    steady_form = ( B_g(a, u, p, v, q) - dot(f, v)*dx )

    # APPLY STABILIZATION
    if parameters.config['stabilization']:
        #turn individual terms on and off by tweaking delta0, tau0
        if delta > 0:
            steady_form += delta*(dot(N(a, u, p) - f, Phi(a, v)))*dx
        if tau > 0:
            steady_form += tau*(dot(div(u), div(v)))*dx

    return steady_form

def build_temperature_form(T, T_n, T_v, u_):
    """Define temperature variational problem to be solved in simulation.

    Parameters
    ----------
    f : dict
        Functions dictionary (as output, for example, by
        feffi.parameters.define_functions())
    mesh : fenics-compatible mesh object
        Mesh to use for simulation
     kwargs : `rho_0`, `nu`, `alpha`, `steps_n`, `g`, `beta`, `gamma`,
        `T_0`, `S_0`.

    Return
    ------
    stiffnes_mats : dict
        Stiffness matrices ready for assembly
    load_vectors : dict
        Load vectors ready for assembly.

    Examples
    --------
    1) Define IPCS variational forms over a square:

        mesh = feffi.mesh.create_mesh(domain='square')
        f_spaces = feffi.functions.define_function_spaces(mesh)
        f = feffi.functions.define_functions(f_spaces)
        feffi.functions.init_functions(f)
        (stiffness_mats, load_vectors) = feffi.functions.define_variational_problems(f, mesh)
    """

    # Assemble tensor viscosity/diffusivity
    alpha = parameters.assemble_viscosity_tensor(parameters.config['alpha']);

    # Define expressions used in variational forms
    dt = 1/parameters.config['steps_n']

    def get_matrix_diagonal(mat):
        diag = []
        for i in range(mat.ufl_shape[0]):
            diag.append(mat[i][i])

        return fenics.as_vector(diag)

    # Variational problem for temperature
    F = + dot((T - T_n)/dt, T_v)*dx \
        + div(u_*T)*T_v*dx \
        + dot(elem_mult(get_matrix_diagonal(alpha), grad(T)), grad(T_v))*dx

    return F

def define_variational_problems(f, mesh, **kwargs):
    """Define variational problems to be solved in simulation.

    We use a modified version of Chorin's method, the so-called
    Incremental Pressure Correction Splitting (IPCS) scheme due to Goda (1979).

    Parameters
    ----------
    f : dict
        Functions dictionary (as output, for example, by
        feffi.parameters.define_functions())
    mesh : fenics-compatible mesh object
        Mesh to use for simulation
     kwargs : `rho_0`, `nu`, `alpha`, `steps_n`, `g`, `beta`, `gamma`,
        `T_0`, `S_0`.

    Return
    ------
    stiffnes_mats : dict
        Stiffness matrices ready for assembly
    load_vectors : dict
        Load vectors ready for assembly.

    Examples
    --------
    1) Define IPCS variational forms over a square:

        mesh = feffi.mesh.create_mesh(domain='square')
        f_spaces = feffi.functions.define_function_spaces(mesh)
        f = feffi.functions.define_functions(f_spaces)
        feffi.functions.init_functions(f)
        (stiffness_mats, load_vectors) = feffi.functions.define_variational_problems(f, mesh)
    """

    # Allow function arguments to overwrite wide config (but keep it local)
    config = dict(parameters.config); config.update(kwargs)

    # Shorthand for functions used in variational forms
    u = f['u']; u_n = f['u_n']; v = f['v']; u_ = f['u_']
    p = f['p']; p_n = f['p_n']; q = f['q']; p_ = f['p_']
    T = f['T']; T_n = f['T_n']; T_v = f['T_v']
    S = f['S']; S_n = f['S_n']; S_v = f['S_v']
    rho_0 = config['rho_0']; g = config['g'];

    # Assemble tensor viscosity/diffusivity
    nu = parameters.assemble_viscosity_tensor(config['nu']);
    alpha = parameters.assemble_viscosity_tensor(config['alpha']);

    # Define expressions used in variational forms
    stiffness_mats = {}; load_vectors = {}
    U = 0.5*(u_n + u)
    n = fenics.FacetNormal(mesh)
    dt = 1/config['steps_n']

    def get_matrix_diagonal(mat):
        diag = []
        for i in range(mat.ufl_shape[0]):
            diag.append(mat[i][i])

        return fenics.as_vector(diag)

    # Define variational problem for approximated velocity
    '''y = fenics.Expression("1-x[1]", degree=2)
    F1 = + dot((u - u_n)/dt, v)*dx \
         + dot(dot(u_n, nabla_grad(u_n)), v)*dx \
         + inner(2*elem_mult(nu, sym(nabla_grad(U))), sym(nabla_grad(v)))*dx \
         - inner((p_n - rho_0*g*y)/rho_0*Identity(len(U)), sym(nabla_grad(v)))*dx \
         + dot((p_n - rho_0*g*y)*n/rho_0, v)*ds \
         - dot(elem_mult(nu, nabla_grad(U))*n, v)*ds \
         - dot(buoyancy, v)*dx
    stiffness_mats['a1'], load_vectors['L1'] = fenics.lhs(F1), fenics.rhs(F1)

    # Variational problem for pressure p with approximated velocity u
    F2 = + dot(nabla_grad(p - p_n), nabla_grad(q))/rho_0*dx \
         + div(u_)*q*(1/dt)*dx
    stiffness_mats['a2'], load_vectors['L2'] = fenics.lhs(F2), fenics.rhs(F2)

    # Variational problem for corrected velocity u with pressure p
    F3 = + dot(u, v)*dx \
         - dot(u_, v)*dx \
         + dot(nabla_grad(p_ - p_n), v)/rho_0*dt*dx
    stiffness_mats['a3'], load_vectors['L3'] = fenics.lhs(F3), fenics.rhs(F3)'''

    # Variational problem for temperature
    F4 = + dot((T - T_n)/dt, T_v)*dx \
         + div(u_*T)*T_v*dx \
         + dot(elem_mult(get_matrix_diagonal(alpha), grad(T)), grad(T_v))*dx
    stiffness_mats['a4'], load_vectors['L4'] = fenics.lhs(F4), fenics.rhs(F4)

    # Variational problem for salinity
    F5 = + dot((S - S_n)/dt, S_v)*dx \
         + div(u_*S)*S_v*dx \
         + dot(elem_mult(get_matrix_diagonal(alpha), grad(S)), grad(S_v))*dx
    stiffness_mats['a5'], load_vectors['L5'] = fenics.lhs(F5), fenics.rhs(F5)

    flog.info('Defined variational problems')

    return stiffness_mats, load_vectors
