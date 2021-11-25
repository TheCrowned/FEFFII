from fenics import (dx, Function, FunctionSpace, FiniteElement, lhs, rhs,
                    Constant, norm, MixedElement, TestFunctions, solve,
                    TrialFunctions, ln)
from . import parameters
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
    mesh = uw.function_space().mesh()
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement([P1, P1, P1])
    V = FunctionSpace(mesh, element)
    v_1, v_2, v_3 = TestFunctions(V)
    mw, Tzd, Szd = TrialFunctions(V)
    sol = Function(V)

    ## Shorthand for constants
    Cd    = parameters.config['3eqs']['Cd']
    cw    = parameters.config['3eqs']['cw']
    rhofw = parameters.config['3eqs']['rhofw']
    rhosw = parameters.config['3eqs']['rhosw']
    lam1  = parameters.config['3eqs']['lam1']
    lam2  = parameters.config['3eqs']['lam2']
    lam3  = parameters.config['3eqs']['lam3']
    L     = parameters.config['3eqs']['L']
    Ut    = parameters.config['3eqs']['Ut']
    rho_I    = parameters.config['3eqs']['rho_I']
    c_pI    = parameters.config['3eqs']['c_pI']
    k_I    = parameters.config['3eqs']['k_I']
    g    = parameters.config['3eqs']['g']
    Ts    = parameters.config['3eqs']['Ts']
    gammaT    = parameters.config['3eqs']['gammaT']
    gammaS    = parameters.config['3eqs']['gammaS']

    Ustar = (Cd*(uw.sub(0)**2+uw.sub(1)**2)+Ut**2)**(1/2) # Ut is needed, maybe in case uw is 0 at some point?

    #Ustar_expr = fenics.interpolate(fenics.Expression('sqrt((Cd*(u1*u1+u2*u2)+Ut*Ut))', degree=2, Cd=Cd, Ut=Ut, u1=uw.sub(0), u2=uw.sub(1)), V.extract_sub_space([1]).collapse())

    #gammaT = fenics.Expression('1/(12.5*pow((nu/alpha), 2/3)-1.12)', degree=2, nu=parameters.config['nu'][1], alpha=parameters.config['alpha'][1])
    #gammaS = gammaT

    y = fenics.interpolate(fenics.Expression('1-x[1]', degree=2), V.extract_sub_space([1]).collapse())
    F = ( (- rhofw*mw*L + rho_I*c_pI*k_I*(Ts-Tzd)/(y) - rhosw*cw*Ustar*gammaT*(Tzd-Tw))*v_1*dx
           + (Tzd - lam1*Szd - lam2 - lam3*pzd)*v_2*dx
           + (rhofw*mw*Sw - rhosw*Ustar*gammaS*(Szd-Sw))*v_3*dx )
           # last equation should have lhs rhofw*mw*Szd, but this is a common
           # linear approximation (source: Johan)

    solve(lhs(F) == rhs(F), sol)
    sol_splitted = sol.split()
    #sol_splitted[0].assign(sol_splitted[0]*)
    sol_splitted[0].rename('mw', 'meltrate')
    sol_splitted[1].rename('Tzd', 'Tzd')
    sol_splitted[2].rename('Szd', 'Szd')
    return (sol_splitted[0], sol_splitted[1], sol_splitted[2])


def build_heat_flux_forcing_term(u_, Tw, mw, Tzd):

    ## Shorthand for constants
    Cd    = parameters.config['3eqs']['Cd']
    cw    = parameters.config['3eqs']['cw']
    rhofw = parameters.config['3eqs']['rhofw']
    rhosw = parameters.config['3eqs']['rhosw']
    Ut    = parameters.config['3eqs']['Ut']
    gammaT    = parameters.config['3eqs']['gammaT']

    #gammaT = fenics.Expression('1/(12.5*pow((nu/alpha), 2/3)-1.12)', degree=2, nu=parameters.config['nu'][1], alpha=parameters.config['alpha'][1])
    #print(gammaT(0.44, 0.55))
    Ustar = (Cd*(u_.sub(0)**2+u_.sub(1)**2)+Ut**2)**(1/2)

    Fh = -(Ustar*3.6*gammaT+mw*3.6)*(Tzd-Tw)

    # These are for to allow plotting of flux boundary term values
    Ustar = fenics.Expression('sqrt((Cd*(u1*u1+u2*u2)+Ut*Ut))', degree=2, Cd=Cd, Ut=Ut, u1=u_.sub(0), u2=u_.sub(1))
    Ustar = fenics.interpolate(Ustar, Tzd.function_space().collapse())
    Fh_func = fenics.Expression('-cw*(Ustar*3.6*gammaT+mw*3.6)*(Tzd-Tw)', degree=2, rhosw=rhosw, Ustar=Ustar, gammaT=gammaT, rhofw=rhofw, Tzd=Tzd, Tw=Tw, mw=mw, cw=cw)
    Fh_func = fenics.interpolate(Fh_func, Tzd.function_space().collapse())
    Fh_func.rename('heat_flux', '')

    return Fh, Fh_func


def build_salinity_flux_forcing_term(u_, Sw, mw, Szd):

    ## Shorthand for constants
    Cd    = parameters.config['3eqs']['Cd']
    rhofw = parameters.config['3eqs']['rhofw']
    rhosw = parameters.config['3eqs']['rhosw']
    Ut    = parameters.config['3eqs']['Ut']
    gammaS    = parameters.config['3eqs']['gammaS']

    #gammaS = fenics.Expression('1/(12.5*pow((nu/alpha), 2/3)-1.12)', degree=2, nu=parameters.config['nu'][1], alpha=parameters.config['alpha'][1])

    Ustar = (Cd*(u_.sub(0)**2+u_.sub(1)**2)+Ut**2)**(1/2)

    Fs = -(Ustar*3.6*gammaS+mw*3.6)*(Szd-Sw)

    # These are for to allow plotting of flux boundary term values
    Ustar = fenics.Expression('sqrt((Cd*(u1*u1+u2*u2)+Ut*Ut))', degree=2, Cd=Cd, Ut=Ut, u1=u_.sub(0), u2=u_.sub(1))
    Ustar = fenics.interpolate(Ustar, Szd.function_space().collapse())
    Fs_func = fenics.Expression('-(Ustar*3.6*gammaS+mw*3.6)*(Szd-Sw)', degree=2, rhosw=rhosw, Ustar=Ustar, gammaS=gammaS, rhofw=rhofw, Szd=Szd, Sw=Sw, mw=mw)
    Fs_func = fenics.interpolate(Fs_func, Szd.function_space().collapse())
    Fs_func.rename('salt_flux', '')

    return Fs, Fs_func