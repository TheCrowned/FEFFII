from fenics import SubDomain, near
import numpy as np
from feffi import parameters

#tolerance = 1000 #High to be sure of catching mistakes if any

class Bound_Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0) and on_boundary

class Bound_Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0) and on_boundary

class Bound_Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], parameters.config['domain_size_x']) and on_boundary

class Bound_Ice_Shelf_Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return ((x[0] >= 0 and x[0] <= parameters.config['shelf_size_x']) and near(x[1], parameters.config['domain_size_y'] - parameters.config['shelf_size_y']) and on_boundary)

class Bound_Ice_Shelf_Right(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], parameters.config['shelf_size_x']) and (x[1] >= parameters.config['domain_size_y'] - parameters.config['shelf_size_y'] and x[1] <= parameters.config['domain_size_y']) and on_boundary)

class Bound_Sea_Top(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] >= parameters.config['shelf_size_x'] and near(x[1], parameters.config['domain_size_y']) and on_boundary)