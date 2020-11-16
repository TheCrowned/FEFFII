from fenics import dot, inner, elem_mult, grad, nabla_grad, div, dx, ds, sym, Identity
import fenics
from . import parameters
import logging
flog = logging.getLogger('feffi')

def define_function_spaces(mesh):
    """Define function spaces for velocity, pressure, temperature and salinity.

    Parameters
    ----------
    mesh : a fenics-compatible mesh object
        Mesh on which to define function spaces.

    Return
    ------
    function_spaces : dictionary
    """

    f_spaces = {
        'V': fenics.VectorFunctionSpace(mesh, 'CG', 2),
        'Q': fenics.FunctionSpace(mesh, 'CG', 1),
        'T': fenics.FunctionSpace(mesh, 'CG', 2),
        'S': fenics.FunctionSpace(mesh, 'CG', 2)
    }

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
        'u_n': fenics.Function(f_spaces['V']),
        'u_': fenics.Function(f_spaces['V']),
        'u': fenics.TrialFunction(f_spaces['V']),
        'v': fenics.TestFunction(f_spaces['V']),
        'p_n': fenics.Function(f_spaces['Q']),
        'p_': fenics.Function(f_spaces['Q']),
        'p': fenics.TrialFunction(f_spaces['Q']),
        'q': fenics.TestFunction(f_spaces['Q']),
        'T_n': fenics.Function(f_spaces['T']),
        'T_': fenics.Function(f_spaces['T']),
        'T': fenics.TrialFunction(f_spaces['T']),
        'T_v': fenics.TestFunction(f_spaces['T']),
        'S_n': fenics.Function(f_spaces['S']),
        'S_': fenics.Function(f_spaces['S']),
        'S': fenics.TrialFunction(f_spaces['S']),
        'S_v': fenics.TestFunction(f_spaces['S'])
    }

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
    f['T_n'].assign(
        fenics.interpolate(
            fenics.Constant(config['T_0']),
            f['T_n'].ufl_function_space()))
    f['S_n'].assign(
        fenics.interpolate(
            fenics.Constant(config['S_0']),
            f['S_n'].ufl_function_space()))
    f['p_n'].assign(
        fenics.interpolate(
            fenics.Expression(
                'rho_0*g*(1-x[1])',
                degree=2,
                rho_0=config['rho_0'],
                g=config['g']),
            f['p_n'].ufl_function_space()))

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
    U = 0.5*(u_n + u)
    n = fenics.FacetNormal(mesh)
    dt = 1/config['steps_n']

    def get_matrix_diagonal(mat):
        diag = []
        for i in range(mat.ufl_shape[0]):
            diag.append(mat[i][i])

        return fenics.as_vector(diag)

    stiffness_mats = {}; load_vectors = {}

    # Define variational problem for approximated velocity
    buoyancy = fenics.Expression(
        (0, '-g*(-beta*(T_ - T_0) + gamma*(S_ - S_0))'),
        beta = config['beta'], gamma = config['gamma'],
        T_0 = config['T_0'], S_0 = config['S_0'],
        g = config['g'],
        T_ = f['T_'], S_ = f['S_'],
        degree=2)
    y = fenics.Expression("1-x[1]", degree=2)
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
    stiffness_mats['a3'], load_vectors['L3'] = fenics.lhs(F3), fenics.rhs(F3)

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