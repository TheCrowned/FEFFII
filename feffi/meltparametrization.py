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
        'gammaT' : 1.15*10**(-4), # this depends on mesh resolution (1.15*10**(-2) for mesh-res 10)
        'gammaS' : 1.15*10**(-4)/35,     # John had gammaS=gammaT/3 cause he says salinity change is not strong enough. ISOMIP paper provides a way to tune these gammas wrt the desired meltrate
        'L' : 3.34*10**5,
        'lam1' : -0.0573,
        'lam2' : 0.0832,
        'lam3' : -7.53*10**(-8),
        'rhofw' : 1000,
        'rhosw' : 1028,
        'Ut' : 0.01,
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
    gammaT    = parameters.config['3eqs']['gammaT']
    gammaS    = parameters.config['3eqs']['gammaS']

    #Ustar = (Cd*norm(uw)**2+Ut**2)**(1/2) # Ustar^2 = Cd(Uw^2 + Ut^2)
    #Ustar = (Cd*uw**2+Ut**2)**(1/2) # Ustar^2 = Cd(Uw^2 + Ut^2) # ustar should be a function
    Ustar = (Cd*(uw.sub(0)**2+uw.sub(1)**2)+Ut**2)**(1/2)        # Ut is needed, maybe in case uw is 0 at some point?
    #gamT = 1/(2.12*ln((Ustar*mesh.hmax())/max(parameters.config['nu']))+12.5*(max(parameters.config['nu'])/max(parameters.config['alpha']))-9) # from other paper

    # should Ustar be in some way only computed on the boundary??
    F = ( (+ rhofw*mw*L + rhosw*cw*Ustar*gammaT*(Tzd-Tw))*v_1*dx
           + (Tzd - lam1*Szd - lam2 - lam3*pzd)*v_2*dx
           + (rhofw*mw*Sw + rhosw*Ustar*gammaS*(Szd-Sw))*v_3*dx )
           # last equation should have lhs rhofw*mw*Szd, but this is a common
           # linear approximation (source: Johan)

    solve(lhs(F) == rhs(F), sol)
    sol_splitted = sol.split()
    return (sol_splitted[0], sol_splitted[1], sol_splitted[2])


def build_heat_flux_forcing_term(u_, Tw, mw, Tzd):

    ## Shorthand for constants
    Cd    = parameters.config['3eqs']['Cd']
    cw    = parameters.config['3eqs']['cw']
    rhofw = parameters.config['3eqs']['rhofw']
    rhosw = parameters.config['3eqs']['rhosw']
    Ut    = parameters.config['3eqs']['Ut']
    gammaT    = parameters.config['3eqs']['gammaT']

    #Ustar = (Cd*norm(u_)**2+Ut**2)**(1/2) # Ustar^2 = Cd(Uw^2 + Ut^2)
    Ustar = (Cd*(u_.sub(0)**2+u_.sub(1)**2)+Ut**2)**(1/2)
    Fh = -cw*(rhosw*Ustar*gammaT+rhofw*mw)*(Tzd-Tw)
    return Fh


def build_salinity_flux_forcing_term(u_, Sw, mw, Szd):

    ## Shorthand for constants
    Cd    = parameters.config['3eqs']['Cd']
    rhofw = parameters.config['3eqs']['rhofw']
    rhosw = parameters.config['3eqs']['rhosw']
    Ut    = parameters.config['3eqs']['Ut']
    gammaS    = parameters.config['3eqs']['gammaS']

    #Ustar = (Cd*norm(u_)**2+Ut**2)**(1/2) # Ustar^2 = Cd(Uw^2 + Ut^2)
    Ustar = (Cd*(u_.sub(0)**2+u_.sub(1)**2)+Ut**2)**(1/2)
    Fs = -(rhosw*Ustar*gammaS+rhofw*mw)*(Szd-Sw)
    return Fs