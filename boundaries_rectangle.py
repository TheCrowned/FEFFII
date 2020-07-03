from fenics import *
import numpy as np

args = {} #High to be sure of catching mistakes if any

class Bound_Left(SubDomain):
	def inside(self, x, on_boundary):
		if(near(x[0], 0, tolerance) and on_boundary):
			return True

		return False

class Bound_Right(SubDomain):
	def inside(self, x, on_boundary):
		if(near(x[0], args.domain_size_x, tolerance) and on_boundary):
			return True

		return False

class Bound_Bottom(SubDomain):
	def inside(self, x, on_boundary):
		if(near(x[1], 0, tolerance) and on_boundary):
			return True

		return False

class Bound_Top(SubDomain):
	def inside(self, x, on_boundary):
		if(near(x[1], args.domain_size_y, tolerance) and on_boundary):
			return True

		return False