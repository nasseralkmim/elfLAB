__author__ = 'Nasser'
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
import matplotlib.pyplot as plt
import gmsh
import element
import assemble
import plotter
import boundaryconditions

mesh = gmsh.parse('mesh4')

ele = element.Matrices(mesh)

s = mesh.surfaces
def k(x1, x2):
    return {s[0]: 1}

ele.stiffness(k)


def load(x1, x2):
    return 0.0


ele.load_internal(load)


K = assemble.globalMatrix(ele.K, mesh)
R = assemble.globalVector(ele.R, mesh)


def traction(x1, x2):
    return {}


def temperature(x1, x2):
    return {0:20,
            1:20,
            2:15,
            3:27,
            4:27,
            5:27}

T = boundaryconditions.neumann(mesh, traction)


B = R + T


K, B = boundaryconditions.dirichlet(K, B, mesh, temperature)

K = sparse.csc_matrix(K)

a = spsolve(K, B)


plotter.trisurface(a, mesh)
plotter.tricontour(a, mesh)
plotter.nodes_network2(mesh)


plt.show()
