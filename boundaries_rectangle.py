from fenics import *
import numpy as np

args = {}
sill = {}
tolerance = 1000 #High to be sure of catching mistakes if any

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
		if(near(x[1], sill['f'](x[0]), tolerance) and (x[0] > sill['left'] and x[0] < sill['right']) and on_boundary):
			return True
		elif(near(x[1], 0) and (x[0] >= sill['right'] or x[0] <= sill['left']) and on_boundary):
			return True

		return False

class Bound_Ice_Shelf_Bottom(SubDomain):
	def inside(self, x, on_boundary):
		if((x[0] >= 0 and x[0] <= args.shelf_size_x) and near(x[1], args.domain_size_y - args.shelf_size_y, tolerance) and on_boundary):
			return True

		return False

class Bound_Ice_Shelf_Right(SubDomain):
	def inside(self, x, on_boundary):
		if(near(x[0], args.shelf_size_x, tolerance) and (x[1] >= args.domain_size_y - args.shelf_size_y and x[1] <= args.domain_size_y) and on_boundary):
			return True

		return False

class Bound_Sea_Top(SubDomain):
	def inside(self, x, on_boundary):
		if(x[0] > args.shelf_size_x and near(x[1], args.domain_size_y, tolerance) and on_boundary):
			return True

		return False
