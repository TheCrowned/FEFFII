from fenics import (dot, inner, elem_mult, grad, nabla_grad, div, cross,
                    dx, ds, sym, Identity, Function, TrialFunction,
                    TestFunction, FunctionSpace, VectorElement, split,
                    FiniteElement, Constant, interpolate, Expression,
                    FacetNormal, as_vector, assemble, norm, MixedElement,
                    TestFunctions, TrialFunctions, solve, lhs, rhs, Measure)
from . import parameters
from .plot import plot_single
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

    V = VectorElement("Lagrange", mesh.ufl_cell(),
                      parameters.config['degree_V'])
    P = FiniteElement("Lagrange", mesh.ufl_cell(),
                      parameters.config['degree_P'])

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

    # If no [TS]_init is given, use [TS]_0
    if(config.get('T_init') == None or len(config['T_init']) == 0):
        config['T_init'] = 'T_0'
    if(config.get('S_init') == None or len(config['S_init']) == 0):
        config['S_init'] = 'S_0'

    f['T_n'].assign(
        interpolate(
            Expression(
                config['T_init'],
                T_0=config['T_0'],  # needed for fallback if T_init is not given
                degree=2),  # would be nice to allow all config variables?,
            f['T_n'].ufl_function_space()))
    f['S_n'].assign(
        interpolate(
            Expression(
                config['S_init'],
                S_0=config['S_0'],
                degree=2),
            f['S_n'].ufl_function_space()))


def N(a, u, p):
    """First stabilization operator, LHS differential operator.
    Corresponds to operator L(U) in LBB paper."""

    dim = a.function_space().mesh().geometric_dimension()
    dt = 1/parameters.config['steps_n']
    nu = parameters.assemble_viscosity_tensor(parameters.config['nu'], dim)
    rho_0 = parameters.config['rho_0']

    return (
        + u/dt
        - div(elem_mult(nu, nabla_grad(u)))
        + dot(a, nabla_grad(u))
        + grad(p)/rho_0)


def Phi(a, u):
    """Second stabilization operator.
    Corresponds to operator Phi in LBB paper."""

    return dot(a, nabla_grad(u))


def B_g(a, u, p_nh, grad_p_h, v, q):
    """Galerkin weak formulation for Navier-Stokes."""

    dim = a.function_space().mesh().geometric_dimension()
    dt = Constant(1/parameters.config['steps_n'])
    nu = parameters.assemble_viscosity_tensor(parameters.config['nu'], dim)
    rho_0 = Constant(parameters.config['rho_0'])
    Omega = Constant((0, 0, parameters.config['Omega_0']))
    n = FacetNormal(a.function_space().mesh())

    # Boundary terms are commented out since we have no Neumann conditions,
    # and Dirichlet conditions would zero them out anyway.
    F = (
        + dot(u, v)/dt*dx
        + (dot(dot(a, nabla_grad(u)), v))*dx
        + inner(elem_mult(nu, nabla_grad(u)), nabla_grad(v))*dx  # sym??
        - dot(p_nh/rho_0, div(v))*dx
        + dot(grad_p_h/rho_0, v)*dx
        # + inner(p_nh*n/rho_0, v)*ds
        # - dot(dot(elem_mult(nu, nabla_grad(u)), n), v)*ds
        - dot(div(u), q)*dx)

    # Add Coriolis acceleration in 3D
    if dim == 3:
        F += dot(cross(2*Omega, u), v)*dx

    return F


def build_buoyancy(T_, S_):
    """Build buoyancy term.

    The degree is governed by pressure degree, since the Expression is added
    to the pressure function."""

    return Expression(
        '-g*(1 - beta*(T_ - T_0) + gamma*(S_ - S_0))',  # g is positive
        beta=Constant(parameters.config['beta']),
        gamma=Constant(parameters.config['gamma']),
        T_0=Constant(parameters.config['T_0']),
        S_0=Constant(parameters.config['S_0']),
        g=Constant(parameters.config['g']),
        T_=T_, S_=S_,
        degree=parameters.config['degree_P'])


def build_NS_GLS_steady_form(a, u, u_n, p, grad_P_h, v, q, T_, S_):
    """Build Navier-Stokes steady state weak form + GLS stabilization."""

    dt = Constant(1/parameters.config['steps_n'])

    # ------------------------
    # Setting stab. parameters
    # ------------------------

    # Init some stuff
    mesh = u.ufl_domain().ufl_cargo()
    (l, k) = (parameters.config['degree_V'], parameters.config['degree_P']) # f spaces degrees
    nu_min = min(parameters.config['nu']) # smallest nu yields biggest Re number -> most unstable
    hmin = mesh.hmin(); hmax = mesh.hmax()
    delta0 = parameters.config['delta0'] #1 # "tuning parameter" > 0
    tau0 = parameters.config['tau0'] # if l == 1 else 0 # "tuning parameter" > 0 dependent on V.degree

    #norm_a = fenics.norm(a) #seems to affect in negative way, and ||a|| is rarely huge
    norm_a = 1

    # Proper definition of stab parameters delta and tau
    Rej = norm_a*hmin/(2*nu_min)
    delta = delta0*hmin*min(1, Rej/3)/norm_a
    tau = tau0*max(nu_min, hmin)

    #b = build_buoyancy(T_, S_)
    f = u_n/dt #+ b
    steady_form = B_g(a, u, p, grad_P_h, v, q) - dot(f, v)*dx

    if parameters.config['stabilization']:
        # Build form
        flog.debug('Stabilized form with Rej = {}; delta = {}; tau = {}'.format(
            round(Rej, 5), round(delta, 5), round(tau, 5)))

        # turn individual terms on and off by tweaking delta0, tau0 in config
        if delta > 0:
            steady_form += delta*(dot(N(a, u, p) - f, Phi(a, v)))*dx
        if tau > 0:
            steady_form += tau*(dot(div(u), div(v)))*dx

    return steady_form


def build_temperature_form(T, T_n, T_v, u_, mw, Tzd, Szd, domain):
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

    mesh = T.function_space().mesh()
    dim = mesh.geometric_dimension()
    alpha = parameters.assemble_viscosity_tensor(parameters.config['alpha'], dim)
    dt = Constant(1/parameters.config['steps_n'])
    n = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=domain.marked_subdomains)

    F = (dot((T - T_n)/dt, T_v)*dx
         + div(u_*T)*T_v*dx
         + dot(elem_mult(get_matrix_diagonal(alpha), grad(T)), grad(T_v))*dx)

    ## (Maybe) Build heat flux forcing term ##
    if mw is not False:
        Cd = 2.5*10**(-3)
        cw = 3974
        gammaT = 1.15*10**(-2)
        rhofw = 1000
        rhosw = 1028
        Ut = 0.01
        Ustar = (Cd*norm(u_)+Ut**2)**(1/2) # np.sqrt(Cd*(np.sqrt(Uw[0]**2 + Uw[1]**2)+Ut**2))            # Ustar^2 = Cd(Uw^2 + Ut^2)
        Tw = T_n

        Fh = -cw*(rhosw*Ustar*gammaT+rhofw*mw)*(Tzd-Tw)

        F += dot(Fh, T_v)*ds(domain.subdomains_markers['left'])

    return F


def build_salinity_form(S, S_n, S_v, u_, mw, Tzd, Szd, domain):
    """Define salinity variational problem.

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

    mesh = S.function_space().mesh()
    dim = mesh.geometric_dimension()
    alpha = parameters.assemble_viscosity_tensor(parameters.config['alpha'], dim)
    dt = Constant(1/parameters.config['steps_n'])
    n = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=domain.marked_subdomains)

    F = (dot((S - S_n)/dt, S_v)*dx
         + div(u_*S)*S_v*dx
         + dot(elem_mult(get_matrix_diagonal(alpha), grad(S)), grad(S_v))*dx)

    ## (Maybe) Build salinity flux forcing term ##
    if mw is not False:
        Cd = 2.5*10**(-3)
        gammaT = 1.15*10**(-2)
        gammaS = gammaT/3
        rhofw = 1000
        rhosw = 1028
        Ut = 0.01
        Ustar = (Cd*norm(u_)+Ut**2)**(1/2) # np.sqrt(Cd*(np.sqrt(Uw[0]**2 + Uw[1]**2)+Ut**2))            # Ustar^2 = Cd(Uw^2 + Ut^2)
        Sw = S_n
        Fh = -(rhosw*Ustar*gammaS+rhofw*mw)*(Szd-Sw)

        F += dot(Fh, S_v)*ds(domain.subdomains_markers['left'])

    return F


def solve_3eqs_system(uw, Tw, Sw, pzd):
    mesh = uw.function_space().mesh()

    # introduce all constants
    # from Asai Davies 2016 ISOMIP Paper
    Cd = 2.5*10**(-3)
    cw = 3974
    gammaT = 1.15*10**(-2)
    gammaS = gammaT/3
    L = 3.34*10**5
    lam1 = -0.0573
    lam2 = 0.0832
    lam3 = -7.53*10**(-8)
    rhofw = 1000
    rhosw = 1028
    Ut = 0.01

    # Ustar from Uw and U_tidal
    Ustar = (Cd*norm(u_)+Ut**2)**(1/2) # np.sqrt(Cd*(np.sqrt(Uw[0]**2 + Uw[1]**2)+Ut**2))            # Ustar^2 = Cd(Uw^2 + Ut^2)

    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement([P1, P1, P1])
    V = FunctionSpace(mesh, element)
    v_1, v_2, v_3 = TestFunctions(V)
    mw, Tzd, Szd = TrialFunctions(V)
    sol = Function(V)

    F = ( (+ rhofw*mw*L + rhosw*cw*Ustar*gammaT*(Tzd-Tw))*v_1*dx
          + Tzd*v_2*dx - lam1*Szd*v_2*dx - lam2*v_2*dx - lam3*pzd*v_2*dx
          + (rhofw*mw*Sw + rhosw*Ustar*gammaS*(Szd-Sw))*v_3*dx
    )
    # why doesn't it work if each term has its own v_1*dx? I get a integration domain missing error.

    solve(lhs(F) == rhs(F), sol)
    sol_splitted = sol.split()

    if norm(sol_splitted[0]) != 0:
        plot_single(sol_splitted[0], display=True, title='meltrate')
    if norm(sol_splitted[1]) != 0:
        plot_single(sol_splitted[1], display=True, title='Tzd')
    if norm(sol_splitted[2]) != 0:
        plot_single(sol_splitted[2], display=True, title='Szd')

    return (sol_splitted[0], sol_splitted[1], sol_splitted[2])


def get_matrix_diagonal(mat):
    diag = [mat[i][i] for i in range(mat.ufl_shape[0])]
    return as_vector(diag)


def energy_norm(u):
    energy = 0.5 * inner(grad(u), grad(u)) * dx
    return assemble(energy)


def get_norms(f):
    d = {}
    for func in ['u', 'p', 'T', 'S']:
        d['||{}||_2'.format(func)] = norm(f[func+'_'], 'L2')
        d['||{}||_inf'.format(func)] = norm(f[func+'_'].vector(), 'linf')
        d['E({})'.format(func)] = energy_norm(f[func+'_'])
    return d