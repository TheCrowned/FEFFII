from fenics import (dx, Function, FunctionSpace, FiniteElement, lhs, rhs,
                    Constant, norm, MixedElement, TestFunctions, solve,
                    TrialFunctions, ln, UserExpression, interpolate, near,
                    Expression)
from . import parameters, plot
import fenics


def get_3eqs_default_constants():
    """Values from Jenkins-Holland 1999 paper."""

    return {
        'c_d' : 1.5*10**(-3),       # momentum exchange coeff
        'c_M' : 3974,               # specific heat capacity of mixed layer
        'c_I' : 2009,               # specific heat capacity of ice
        #'gammaT' : 1*10**(-4),     # these are values at stability, already including u*
        #'gammaS' : 5.05*10**(-7),
        'L' : 3.34*10**5,           # latent heat of fusion
        'a' : -0.0573,              # salinity coeff of freezing eq
        'b' : 0.0939, # 0.0832 misomip # constant coeff of freezing eq
        'c' : -7.53*10**(-8),       # pressure coeff of freezing eq
        'rho_M' : 1025,             # density of mixed layer
        'rho_I' : 920,              # density of ice
        'k_I' : 1.14*10**(-6),      # molecular thermal conductivity ice shelf
        'Ts' : -20,                 # temperature at ice shelf surface (value from mitgcm, JH has -25)
        'Ut' : 0.1,               # tidal velocity, for us a regularization parameter

        ## Gamma(T,S)-related values
        'xi_N' : 0.052,              # stability constant
        'etaStar' : 1,              # stability constant
        'k' : 0.40,                 # Von Kàrmàn's constant
        'Pr' : 13.8,                # Prandtl number
        'Sc' : 2432                 # Schmidt number
    }


def solve_3eqs_system(f):
    """
    Solves the 3 equations system for melt-rate formulation.

    (Notation and equations found in Jenkins-Holland 99 paper:
    https://journals.ametsoc.org/view/journals/phoc/29/8/1520-0485_1999_029_1787_mtioia_2.0.co_2.xml)

    Parameters
    ----------
    f : FEFFI functions dict

    Return
    ------
    m_B : meltrate at ice-ocean boundary
    T_B : temperature field at ice-ocean boundary
    S_B : salinity field at ice-ocean boundary
    """

    # Shorthand for functions
    u_M, T_M, S_M, p_B = f['u_'], f['T_'], f['S_'], f['p_']
    v_1, v_2, v_3 = f['3eqs']['v_m'], f['3eqs']['v_T'], f['3eqs']['v_S']
    m_B, T_B, S_B = f['3eqs']['m_B'], f['3eqs']['T_B'], f['3eqs']['S_B']
    uStar = f['3eqs']['uStar']

    ## Shorthand for constants
    config  = parameters.config
    c_I     = config['3eqs']['c_I']
    c_M     = config['3eqs']['c_M']
    rho_I   = config['3eqs']['rho_I']
    rho_M   = config['3eqs']['rho_M']
    a       = config['3eqs']['a']
    b       = config['3eqs']['b']
    c       = config['3eqs']['c']
    L       = config['3eqs']['L']
    k_I     = config['3eqs']['k_I']
    Ts      = config['3eqs']['Ts']
    xi_N    = config['3eqs']['xi_N']
    etaStar = config['3eqs']['etaStar']
    k       = config['3eqs']['k']
    Pr      = config['3eqs']['Pr']
    Sc      = config['3eqs']['Sc']

    gammaT = Constant(1/(1/(2*xi_N*etaStar)-1/k  +  12.5*(Pr**(2/3)) - 6))
    gammaS = Constant(1/(1/(2*xi_N*etaStar)-1/k  +  12.5*(Sc**(2/3)) - 6))
    #print('gammaT, S {} {}'.format(gammaT.values(), gammaS.values()))

    y = interpolate(Expression('1-x[1]', degree=2), T_M.function_space()) #questionable choice, to divide by this
    F = ( (+ rho_I*m_B*L - rho_I*c_I*k_I*(Ts-T_B)/y + rho_M*c_M*uStar*gammaT*(T_B-T_M))*v_1*dx
           + (T_B - a*S_B - b - c*p_B)*v_2*dx
           + (rho_I*m_B*S_M + rho_M*uStar*gammaS*(S_B-S_M))*v_3*dx )
           # last equation should have lhs rho_I*m_B*S_B, but this is a common
           # linear approximation (source: Johan)

    solve(lhs(F) == rhs(F), f['3eqs']['sol'])
    #(m_B_sol, T_B_sol, S_B_sol) = f['3eqs']['sol'].split()
    #m_B_sol.rename('m_B', 'meltrate')
    #T_B_sol.rename('T_B', 'T_B')
    #S_B_sol.rename('S_B', 'S_B')
    #return (m_B_sol, T_B_sol, S_B_sol)


def build_heat_flux_forcing_term(f):

    # Shorthand for functions
    T_M = f['T_']
    uStar = f['3eqs']['uStar']
    (m_B, T_B, _) = f['3eqs']['sol'].split()

    ## Shorthand for constants
    config  = parameters.config
    xi_N    = config['3eqs']['xi_N']
    etaStar = config['3eqs']['etaStar']
    k       = config['3eqs']['k']
    Pr      = config['3eqs']['Pr']
    Sc      = config['3eqs']['Sc']

    gammaT = Constant(1/(1/(2*xi_N*etaStar)-1/k  +  12.5*(Pr**(2/3)) - 6))

    Fh = -(uStar*gammaT+m_B)*(T_B-T_M)

    # These are for to allow plotting of flux boundary term values
    Fh_func = Expression('-(uStar*gammaT+m_B)*(T_B-T_M)*tanh(1*(x[0]-5))', degree=2, uStar=uStar, gammaT=gammaT, T_B=T_B, T_M=T_M, m_B=m_B)
    Fh_func = interpolate(Fh_func, T_B.function_space().collapse())
    #plot.plot_single(Fh_func, display=True)
    #Fh_func.rename('heat_flux', '')

    return Fh, Fh_func


def build_salinity_flux_forcing_term(f):

    # Shorthand for functions
    S_M = f['S_']
    uStar = f['3eqs']['uStar']
    (m_B, _, S_B) = f['3eqs']['sol'].split()

    ## Shorthand for constants
    config  = parameters.config
    xi_N    = config['3eqs']['xi_N']
    etaStar = config['3eqs']['etaStar']
    k       = config['3eqs']['k']
    Pr      = config['3eqs']['Pr']
    Sc      = config['3eqs']['Sc']

    gammaS = Constant(1/(1/(2*xi_N*etaStar)-1/k  +  12.5*(Sc**(2/3)) - 6))

    Fs = -(uStar*gammaS+m_B)*(S_B-S_M)

    # These are for to allow plotting of flux boundary term values
    Fs_func = Expression('-(uStar*gammaS+m_B)*(S_B-S_M)*tanh(1*(x[0]-5))', degree=2, uStar=uStar, gammaS=gammaS, S_B=S_B, S_M=S_M, m_B=m_B)
    Fs_func = interpolate(Fs_func, S_B.function_space().collapse())
    #plot.plot_single(Fs_func, display=True)
    #Fs_func.rename('salt_flux', '')

    return Fs, Fs_func


class uStar_expr(UserExpression):
    def __init__(self, u):
        self.u = u
        super().__init__()

    def eval(self, value, x):
        ## Shorthand for constants
        config = parameters.config
        ice_shelf_bottom_p = config['ice_shelf_bottom_p']
        ice_shelf_top_p = config['ice_shelf_top_p']
        ice_shelf_slope = config['ice_shelf_slope']
        boundary_layer_thickness = config['boundary_layer_thickness']
        c_d = config['3eqs']['c_d']
        Ut = config['3eqs']['Ut']

        x_b = x[0]
        try:
            y_b = ice_shelf_slope*x[0]+ice_shelf_bottom_p[1]
        except TypeError: # vertical ice shelf
            x_b = 0
            y_b = x[1]

        # Only compute u* for points within boundary_layer_thickness distance from ice shelf
        #if x[0] < ice_shelf_top_p[0] and ice_shelf_slope*x[0]+ice_shelf_bottom_p[1] - x[1] <= boundary_layer_thickness:
        if x[0] <= ice_shelf_top_p[0]+0.05 and y_b - x[1] <= boundary_layer_thickness:
            value[0] = max(10**(-3), c_d*((self.u(x[0], x[1])[0]**2+self.u(x[0], x[1])[1]**2))**(1/2))
        else:
            value[0] = 10**(-3) #c_d*Ut**2

    def value_shape(self):
        return (2,) # cause if value is a scalar, it is not mutable and thus not working
