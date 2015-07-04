__author__ = 'Nasser'
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
import matplotlib.pyplot as plt
import gmsh
import element_1dof
import assemble_1dof
import plotter
import boundaryconditions_1dof

mesh = gmsh.parse('mesh14')


ele = element_1dof.Matrices(mesh)

s = mesh.surfaces
def k(x1, x2):
    return {s[0]: 1}

ele.stiffness(k)


def load(x1, x2):
    return 0.0


ele.load(load)


K = assemble_1dof.globalMatrix(ele.K, mesh)
R = assemble_1dof.globalVector(ele.R, mesh)


def traction(x1, x2):
    return {}


def temperature(x1, x2):
    return {0:20,
            1:20,
            2:15,
            3:27,
            4:27,
            5:27}

T = boundaryconditions_1dof.neumann(mesh, traction)


B = R + T


K, B = boundaryconditions_1dof.dirichlet(K, B, mesh, temperature)

K = sparse.csc_matrix(K)

d = spsolve(K, B)


plotter.trisurface(d, mesh, dpi=60)
plotter.tricontour(d, mesh, name='Temperature', cmap='hot', dpi=60)
#plotter.nodes_network(mesh)
plotter.nodes_network_edges(mesh, dpi=60)
plt.show()
