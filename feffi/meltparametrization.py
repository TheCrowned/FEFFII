from fenics import (dx, Function, FunctionSpace, FiniteElement, lhs, rhs,
                    Constant, norm, MixedElement, TestFunctions, solve,
                    TrialFunctions)

# introduce all constants
# from Asai Davies 2016 ISOMIP Paper
Cd = Constant(2.5*10**(-3))
cw = Constant(3974)
gammaT = Constant(1.15*10**(-2))
gammaS = Constant(gammaT/35) # John had gammaS=gammaT/3 cause he says salinity change is not strong enough. ISOMIP paper provides a way to tune these gammas wrt the desired meltrate
L = Constant(3.34*10**5)
lam1 = Constant(-0.0573)
lam2 = Constant(0.0832)
lam3 = Constant(-7.53*10**(-8))
rhofw = Constant(1000)
rhosw = Constant(1028)
Ut = Constant(0.01)

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
    pzdw : FEniCS Function
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

    Ustar = (Cd*norm(uw)**2+Ut**2)**(1/2) # Ustar^2 = Cd(Uw^2 + Ut^2)
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
    Ustar = (Cd*norm(u_)**2+Ut**2)**(1/2) # Ustar^2 = Cd(Uw^2 + Ut^2)
    Fh = -cw*(rhosw*Ustar*gammaT+rhofw*mw)*(Tzd-Tw)
    return Fh


def build_salinity_flux_forcing_term(u_, Sw, mw, Szd):
    Ustar = (Cd*norm(u_)**2+Ut**2)**(1/2) # Ustar^2 = Cd(Uw^2 + Ut^2)
    Fs = -(rhosw*Ustar*gammaS+rhofw*mw)*(Szd-Sw)
    return Fs