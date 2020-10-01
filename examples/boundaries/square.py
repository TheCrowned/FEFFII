from fenics import SubDomain, near

tolerance = 1000 #High to be sure of catching mistakes if any

class Square_Bound_Top(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[1], 1) and on_boundary

class Square_Bound_Bottom(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[1], 0) and on_boundary

class Square_Bound_Left(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], 0) and on_boundary

class Square_Bound_Right(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], 1) and on_boundary