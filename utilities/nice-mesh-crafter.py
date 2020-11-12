from fenics import *
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(10,10)

for thresh in [0.3, 0.2, 0.1, 0.025]:
    class Bound_Top(SubDomain):
        def inside(self, x, on_boundary):
            return x[1] >= 1-thresh and x[0] <= 1-thresh
    class Bound_Right(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] >= 1-thresh and x[1] >= thresh
    class Bound_Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return x[1] <= thresh and x[0] >= thresh
    class Bound_Left(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] <= thresh and x[1] <= 1-thresh

    boundary_domain = MeshFunction("bool", mesh, mesh.topology().dim() - 1)
    boundary_domain.set_all(False)

    boundaries = [Bound_Top, Bound_Right, Bound_Bottom, Bound_Left]
    for boundary in boundaries:
        obj = boundary()
        obj.mark(boundary_domain, True)

    mesh = refine(mesh, boundary_domain)
    print(mesh.hmin())
    print(mesh.hmax())
    print(mesh.num_vertices())
    plot(mesh)
    plt.show()