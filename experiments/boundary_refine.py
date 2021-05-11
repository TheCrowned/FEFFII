from __future__ import print_function
from datetime import datetime
from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import sys, getopt, argparse
import math

mesh = UnitSquareMesh(12,12)

print(SubDomain)
class Inflow(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] >= 0.5 -DOLFIN_EPS
        
sub_domains_bool = MeshFunction("bool", mesh, mesh.topology().dim() - 1)
sub_domains_bool.set_all(False)
inflow=Inflow()
inflow.mark(sub_domains_bool, True)

mesh = refine(mesh, sub_domains_bool)

plot(mesh)
plt.show()
