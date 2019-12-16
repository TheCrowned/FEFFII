from fenics import *
import numpy as np

tolerance = 1000 #High to be sure of catching mistakes if any

class Bound_Left(SubDomain):
	def inside(self, x, on_boundary):
		if(near(x[0], 0, tolerance) and on_boundary):
			return True

		return False
		
class Bound_Right(SubDomain):
	def inside(self, x, on_boundary):
		if(near(x[0], 1, tolerance) and on_boundary):
			return True

		return False

class Bound_Bottom(SubDomain):
	def inside(self, x, on_boundary):
		if(near(x[1], 0.1*np.sin(3*pi*(x[0]-(3/8))), tolerance) and (x[0] > 0.3 and x[0] < 0.75) and on_boundary):
			return True
		elif(near(x[1], 0) and (x[0] >= 0.75 or x[0] <= 0.3) and on_boundary):
			return True
		
		return False

class Bound_Ice_Shelf_Bottom(SubDomain):
	def inside(self, x, on_boundary):
		if((x[0] >= 0 and x[0] <= 0.4) and near(x[1], 0.9, tolerance) and on_boundary):
			return True
		
		return False

class Bound_Ice_Shelf_Right(SubDomain):
	def inside(self, x, on_boundary):
		if(near(x[0], 0.4, tolerance) and (x[1] >= 0.9 and x[1] <= 1) and on_boundary):
			return True
		
		return False

class Bound_Sea_Top(SubDomain):
	def inside(self, x, on_boundary):
		if(x[0] > 0.4 and near(x[1], 1, tolerance) and on_boundary):
			return True
			
		return False

'''
class Bound_Top(SubDomain):
	def inside(x, on_boundary):
		if(ice_shelf_bottom(x, on_boundary) or ice_shelf_right(x, on_boundary) or sea_top(x, on_boundary)):
			return True

		return False
'''
