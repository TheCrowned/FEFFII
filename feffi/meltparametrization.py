from fenics import (dx, Function, FunctionSpace, FiniteElement, lhs, rhs,
                    Constant, norm, MixedElement, TestFunctions, solve,
                    TrialFunctions, ln, UserExpression, interpolate, near)
from . import parameters, plot
import fenics


def get_3eqs_default_constants():
    """Values from Asai Davies 2016 ISOMIP Paper."""

    return {
        'Cd' : 2.5*10**(-3),
        'cw' : 3974,
        'gammaT' : 1*10**(-4), # this depends on mesh resolution (1.15*10**(-2) for mesh-res 10)
        'gammaS' : 5.05*10**(-7),
        'L' : 3.34*10**5,
        'lam1' : -0.0573,
        'lam2' : 0.0939, # 0.0832 misomip
        'lam3' : -7.53*10**(-8),
        'rhofw' : 1000,
        'rhosw' : 1028,
        'rho_I' : 920,
        'c_pI' : 2009,
        'k_I' : 1.14*10**(-6),
        'Ts' : -20,
        'g' : 9.81,
        'Ut' : 0.1,
    }


def solve_3eqs_system(uw, Tw, Sw, pzd):
    """
    Solves the 3 equations system for melt-rate formulation.

    (Notation and equations found in ISOMIP paper:
    https://gmd.copernicus.org/articles/9/2471/2016/ )

    Parameters
    ----------
    uw : FEniCS Function
        Ocean velocity
    Tw : FEniCS Function
        Ocean temperature
    Sw : FEniCS Function
        Ocean salinity
    pzd : FEniCS Function
        Ocean pressure

    Return
    ------
    mw : meltrate
    Tzd : temperature field at ice-ocean interface
    Szd : salinity field at ice-ocean interface
    """
    config = parameters.config
    mesh = uw.function_space().mesh()
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement([P1, P1, P1])
    V = FunctionSpace(mesh, element)
    v_1, v_2, v_3 = TestFunctions(V)
    mw, Tzd, Szd = TrialFunctions(V)
    sol = Function(V)

    ## Shorthand for constants
    #Cd    = config['3eqs']['Cd']
    cw    = config['3eqs']['cw']
    rhofw = config['3eqs']['rhofw']
    rhosw = config['3eqs']['rhosw']
    lam1  = config['3eqs']['lam1']
    lam2  = config['3eqs']['lam2']
    lam3  = config['3eqs']['lam3']
    L     = config['3eqs']['L']
    rho_I = config['3eqs']['rho_I']
    c_pI  = config['3eqs']['c_pI']
    k_I   = config['3eqs']['k_I']
    Ts    = config['3eqs']['Ts']
    #gammaT    = config['3eqs']['gammaT']
    #gammaS    = config['3eqs']['gammaS']

    Ustar = interpolate(Ustar_expr(uw), uw.function_space()).sub(0)
    #plot.plot_single(Ustar, display=True, file_name='ustar.png')

    #gammaT = fenics.Expression('1/(2.12*std::log(Ustar*0.03/nu)+12.5*pow((nu/alpha), 2/3)-1.12)', degree=2, Ustar=Ustar, nu=parameters.config['nu'][1], alpha=parameters.config['alpha'][1])
    #plot.plot_single(fenics.interpolate(gammaT,V.extract_sub_space([1]).collapse()), display=True)
    #gammaS = gammaT

    ## gamma(T,S)
    xiN = 0.052
    etaStar = 1
    k = 0.40
    Pr = 13.8
    Sc = 2432

    gammaT = fenics.Constant(1/(1/(2*xiN*etaStar)-1/k  +  12.5*(Pr**(2/3)) - 6)) #', degree=2, xiN=xiN, etaStar=etaStar, )
    gammaS = fenics.Constant(1/(1/(2*xiN*etaStar)-1/k  +  12.5*(Sc**(2/3)) - 6)) #', degree=2, xiN=xiN, etaStar=etaStar, )
    #print('gammaT, S {} {}'.format(gammaT.values(), gammaS.values()))

    y = interpolate(fenics.Expression('1-x[1]', degree=2), V.extract_sub_space([1]).collapse())
    F = ( (+ rhofw*mw*L - rho_I*c_pI*k_I*(Ts-Tzd)/y + rhosw*cw*Ustar*gammaT*(Tzd-Tw))*v_1*dx
           + (Tzd - lam1*Szd - lam2 - lam3*pzd)*v_2*dx
           + (rhofw*mw*Sw + rhosw*Ustar*gammaS*(Szd-Sw))*v_3*dx )
           # last equation should have lhs rhofw*mw*Szd, but this is a common
           # linear approximation (source: Johan)

    solve(lhs(F) == rhs(F), sol)
    sol_splitted = sol.split()
    sol_splitted[0].rename('mw', 'meltrate')
    sol_splitted[1].rename('Tzd', 'Tzd')
    sol_splitted[2].rename('Szd', 'Szd')
    #plot.plot_single(Tw, display=True)
    #plot.plot_single(sol_splitted[0], display=True)
    #plot.plot_single(sol_splitted[1], display=True)
    #plot.plot_single(sol_splitted[2], display=True)
    return (sol_splitted[0], sol_splitted[1], sol_splitted[2])


def build_heat_flux_forcing_term(uw, Tw, mw, Tzd):

    ## Shorthand for constants
    Cd    = parameters.config['3eqs']['Cd']
    cw    = parameters.config['3eqs']['cw']
    rhofw = parameters.config['3eqs']['rhofw']
    rhosw = parameters.config['3eqs']['rhosw']
    Ut    = parameters.config['3eqs']['Ut']
    gammaT    = parameters.config['3eqs']['gammaT']

    #gammaT = fenics.Expression('1/(12.5*pow((nu/alpha), 2/3)-1.12)', degree=2, nu=parameters.config['nu'][1], alpha=parameters.config['alpha'][1])
    #print(gammaT(0.44, 0.55))
    Ustar = interpolate(Ustar_expr(uw), uw.function_space()).sub(0)


    ## gamma(T,S)
    xiN = 0.052
    etaStar = 1
    k = 0.40
    Pr = 13.8
    Sc = 2432

    gammaT = fenics.Constant(1/(1/(2*xiN*etaStar)-1/k  +  12.5*(Pr**(2/3)) - 6)) #', degree=2, xiN=xiN, etaStar=etaStar, )
    gammaS = fenics.Constant(1/(1/(2*xiN*etaStar)-1/k  +  12.5*(Sc**(2/3)) - 6)) #', degree=2, xiN=xiN, etaStar=etaStar, )

    Fh = -(Ustar*gammaT+mw)*(Tzd-Tw)

    # These are for to allow plotting of flux boundary term values
    #Ustar = fenics.Expression('sqrt((Cd*(u1*u1+u2*u2)+Ut*Ut))', degree=2, Cd=Cd, Ut=Ut, u1=uw.sub(0), u2=uw.sub(1))
    #Ustar = fenics.interpolate(Ustar, Tzd.function_space().collapse())
    #Fh_func = fenics.Expression('-(Ustar*gammaT+mw)*(Tzd-Tw)', degree=2, rhosw=rhosw, Ustar=Ustar, gammaT=gammaT, Tzd=Tzd, Tw=Tw, mw=mw)
    #Fh_func = fenics.interpolate(Fh_func, Tzd.function_space().collapse())
    #plot.plot_single(Fh_func, display=True)
    #Fh_func.rename('heat_flux', '')

    return Fh, False


def build_salinity_flux_forcing_term(uw, Sw, mw, Szd):

    ## Shorthand for constants
    Cd    = parameters.config['3eqs']['Cd']
    rhofw = parameters.config['3eqs']['rhofw']
    rhosw = parameters.config['3eqs']['rhosw']
    Ut    = parameters.config['3eqs']['Ut']
    gammaS    = parameters.config['3eqs']['gammaS']

    #gammaS = fenics.Expression('1/(12.5*pow((nu/alpha), 2/3)-1.12)', degree=2, nu=parameters.config['nu'][1], alpha=parameters.config['alpha'][1])

    Ustar = interpolate(Ustar_expr(uw), uw.function_space()).sub(0)


    ## gamma(T,S)
    xiN = 0.052
    etaStar = 1
    k = 0.40
    Pr = 13.8
    Sc = 2432

    gammaT = fenics.Constant(1/(1/(2*xiN*etaStar)-1/k  +  12.5*(Pr**(2/3)) - 6)) #', degree=2, xiN=xiN, etaStar=etaStar, )
    gammaS = fenics.Constant(1/(1/(2*xiN*etaStar)-1/k  +  12.5*(Sc**(2/3)) - 6)) #', degree=2, xiN=xiN, etaStar=etaStar, )

    Fs = -(Ustar*gammaS+mw)*(Szd-Sw)

    # These are for to allow plotting of flux boundary term values
    #Ustar = fenics.Expression('sqrt((Cd*(u1*u1+u2*u2)+Ut*Ut))', degree=2, Cd=Cd, Ut=Ut, u1=uw.sub(0), u2=uw.sub(1))
    #Ustar = fenics.interpolate(Ustar, Szd.function_space().collapse())
    #Fs_func = fenics.Expression('-(Ustar*gammaS+mw)*(Szd-Sw)', degree=2, rhosw=rhosw, Ustar=Ustar, gammaS=gammaS, rhofw=rhofw, Szd=Szd, Sw=Sw, mw=mw)
    #Fs_func = fenics.interpolate(Fs_func, Szd.function_space().collapse())
    #Fs_func.rename('salt_flux', '')

    return Fs, False


class Ustar_expr(UserExpression):
    def __init__(self, u):
        self.u = u
        super().__init__()

    def eval(self, value, x):
        ## Shorthand for constants
        ice_shelf_bottom_p = parameters.config['ice_shelf_bottom_p']
        ice_shelf_top_p = parameters.config['ice_shelf_top_p']
        ice_shelf_slope = parameters.config['ice_shelf_slope']
        boundary_layer_thickness = parameters.config['boundary_layer_thickness']
        Cd = parameters.config['3eqs']['Cd']
        Ut = parameters.config['3eqs']['Ut']

        '''x_b = (x[1]-ice_shelf_bottom_p[1])/ice_shelf_slope
        y_b = x[1]-boundary_layer_thickness
        if y_b <= 0:
            y_b = 0
        if x_b <= 0:
            x_b = 0'''

        #if x[0] < ice_shelf_top_p[0] and ice_shelf_slope*x[0]+ice_shelf_bottom_p[1] - x[1] <= boundary_layer_thickness:
        if x[0] <= ice_shelf_top_p[0]+0.05 and ice_shelf_slope*x[0]+ice_shelf_bottom_p[1] - x[1] <= boundary_layer_thickness:
            value[0] = Cd*((self.u(x[0], x[1])[0]**2+self.u(x[0], x[1])[1]**2) + Ut**2)**(1/2)
        else:
            value[0] = Cd*Ut**2
        #print('p ({}, {}) becomes ({}, {}), val {}'.format(round(x[0],2), round(x[1],2), round(x_b,2), round(y_b,2), round(value[0],5)))

    def value_shape(self):
        return (2,) # cause if value is a scalar, it is not mutable and thus not working