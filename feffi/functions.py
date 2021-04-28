from fenics import (dot, inner, elem_mult, grad, nabla_grad, div,
                    dx, ds, sym, Identity, Function, TrialFunction,
                    TestFunction, FunctionSpace, VectorElement, split,
                    FiniteElement, Constant, interpolate, Expression)
import fenics
from . import parameters
import logging
flog = logging.getLogger('feffi')

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

    # Define functions needed for solution computation
    f = {
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
    (f['v'], f['q']) = split(TestFunction(f_spaces['W']))
    f['sol'] = Function(f_spaces['W'])
    (f['u_'], f['p_']) = f['sol'].split(True)
    (f['u_n'], f['p_n']) = f['sol'].split(True)

    # Nice names for output (ex. in Paraview)
    f['u_'].rename("velocity", "Velocity in m/s")
    f['p_'].rename("pressure", "Pressure in Pa")
    f['T_'].rename("temperature", "Temperature in °C")
    f['S_'].rename("salinity", "Salinity in PSU")

    return f

def init_functions(f, **kwargs):
    """Set function values to closest stable state to speed up convergence.

    Parameters
    ----------
    f : dict
        Functions to initialize
    kwargs : `T_0`, `S_0`, `rho_0`, `g`, `T_init`, `S_init` (refer to README for info).
    """

    # Allow function arguments to overwrite wide config (but keep it local)
    config = dict(parameters.config); config.update(kwargs)

    if(config.get('T_init') == None or len(config['T_init']) == 0):
        config['T_init'] = 'T_0'

    if(config.get('S_init') == None or len(config['S_init']) == 0):
        config['S_init'] = 'S_0'

    f['T_n'].assign(
        interpolate(
            Expression(
                config['T_init'],
                T_0 = config['T_0'], # needed for fallback if T_init is not given
                degree = 2), # would be nice to allow all config variables?,
            f['T_n'].ufl_function_space()))
    f['S_n'].assign(
        interpolate(
            Expression(
                config['S_init'],
                S_0 = config['S_0'],
                degree = 2),
            f['S_n'].ufl_function_space()))

    #It makes no sense to init p without splitting scheme?
    '''f['p_n'].assign(
        interpolate(
            Expression(
                'rho_0*g*(1-x[1])',
                degree=2,
                rho_0=config['rho_0'],
                g=config['g']),
            f['p_n'].ufl_function_space().collapse()))'''

def N(a, u, p):
    """First stabilization operator, LHS differential operator.
    Corresponds to operator L(U) in LBB paper."""

    dt = 1/parameters.config['steps_n']
    nu = parameters.assemble_viscosity_tensor(parameters.config['nu'])
    rho_0 = parameters.config['rho_0']

    return u/dt - div(elem_mult(nu, nabla_grad(u))) + dot(a, nabla_grad(u)) + grad(p)/rho_0

def Phi(a, u):
    """Second stabilization operator.
    Corresponds to operator Phi in LBB paper."""

    return dot(a, nabla_grad(u))

def B_g(a, u, p, grad_P_h, v, q):
    """Galerkin weak formulation for Navier-Stokes."""

    dt = Constant(1/parameters.config['steps_n'])
    nu = parameters.assemble_viscosity_tensor(parameters.config['nu'])
    rho_0 = Constant(parameters.config['rho_0'])
    n = fenics.FacetNormal(a.function_space().mesh())

    return (
    + dot(u, v)/dt*dx
    + inner(elem_mult(nu, nabla_grad(u)), nabla_grad(v))*dx # sym??
    + (dot(dot(a, nabla_grad(u)), v) )*dx
    - dot(p/rho_0, div(v))*dx
    + dot(grad_P_h/rho_0, v)*dx
    + dot(p/rho_0, dot(v, n))*ds
    - dot(dot(elem_mult(nu, nabla_grad(u)), n), v)*ds
    - dot(div(u), q)*dx )

def build_buoyancy(T_, S_):
    """Build buoyancy term."""

    return Expression(
        (0, '-g*(1 - beta*(T_ - T_0) + gamma*(S_ - S_0))'), # g is given positive
        beta = Constant(parameters.config['beta']),
        gamma = Constant(parameters.config['gamma']),
        T_0 = Constant(parameters.config['T_0']),
        S_0 = Constant(parameters.config['S_0']),
        g = Constant(parameters.config['g']),
        T_ = T_, S_ = S_,
        degree=2)

def build_NS_GLS_steady_form(a, u, u_n, p, grad_P_h, v, q, T_, S_):
    """Build Navier-Stokes steady state weak form + GLS stabilization."""

    dt = Constant(1/parameters.config['steps_n'])

    # ------------------------
    # Setting stab. parameters
    # ------------------------

    # Init some stuff
    mesh = u.ufl_domain().ufl_cargo()
    (l, k) = (parameters.config['degree_V'], parameters.config['degree_P']) # f spaces degrees
    nu_min = min(parameters.config['nu']) #smallest nu yields biggest Re number -> most unstable
    hmin = mesh.hmin(); hmax = mesh.hmax()
    delta0 = Constant(parameters.config['delta0']) #1 # "tuning parameter" > 0
    tau0 = Constant(parameters.config['tau0']) # if l == 1 else 0 # "tuning parameter" > 0 dependent on V.degree

    #norm_a = fenics.norm(a) #seems to affect in negative way, and ||a|| is rarely huge
    norm_a = 1;

    # Proper definition of stab parameters delta and tau
    Rej = norm_a*hmin/(2*nu_min)
    delta = delta0*hmin*min(1, Rej/3)/norm_a
    tau = tau0*max(nu_min, hmin)

    # Build form
    flog.debug('Stabilized form with Rej = {}; delta = {}; tau = {}'.format(
        round(Rej, 5), round(delta, 5), round(tau, 5)))

    b = build_buoyancy(T_, S_)
    f = u_n/dt #+ b
    steady_form = B_g(a, u, p, grad_P_h, v, q) - dot(f, v)*dx

    if parameters.config['stabilization']:
        #turn individual terms on and off by tweaking delta0, tau0 in config
        if delta > 0:
            steady_form += delta*(dot(N(a, u, p) - f, Phi(a, v)))*dx
        if tau > 0:
            steady_form += tau*(dot(div(u), div(v)))*dx

        flog.debug('Stabilization terms added to variational form')
    else:
        flog.debug('NO Stabilization terms added to variational form')

    return steady_form

def build_temperature_form(T, T_n, T_v, u_):
    """Define temperature variational problem.

    Parameters
    ----------
    T : FEniCS TrialFunction
    T_n : FEniCS Function with previously computed temperature
    T_v : FEniCS TestFunction
    u_ : FEniCS Function with previously computed velocity field

    Return
    ------
    FEniCS Form
    """

    alpha = parameters.assemble_viscosity_tensor(parameters.config['alpha']);
    dt = 1/parameters.config['steps_n']

    return ( dot((T - T_n)/dt, T_v)*dx
           + div(u_*T)*T_v*dx
           + dot(elem_mult(get_matrix_diagonal(alpha), grad(T)), grad(T_v))*dx )

def build_salinity_form(S, S_n, S_v, u_):
    """Define salinity variational problem.
    Calls build_temperature_form with salinity variables inside.

    Parameters
    ----------
    S : FEniCS TrialFunction
    S_n : FEniCS Function with previously computed salinity
    S_v : FEniCS TestFunction
    u_ : FEniCS Function with previously computed velocity field

    Return
    ------
    FEniCS Form
    """

    return build_temperature_form(S, S_n, S_v, u_)

def get_matrix_diagonal(mat):
    diag = [mat[i][i] for i in range(mat.ufl_shape[0])]
    return fenics.as_vector(diag)