__author__ = 'Nasser'

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
import matplotlib.pyplot as plt
import gmsh
import element_2dof
import assemble_2dof
import plotter
import boundaryconditions_2dof

mesh = gmsh.parse('mesh4')

ele = element_2dof.Matrices(mesh)

s = mesh.surfaces

ele.stiffness(nu=0.1, E=1000.0)

def distributed_load(x1, x2):
    return np.array([
        150.0,
        0.0
    ])

ele.load(distributed_load)

K = assemble_2dof.globalMatrix(ele.K, mesh)
R = assemble_2dof.globalVector(ele.R, mesh)


def traction(x1, x2):
    return {}


def displacement(x1, x2):
    return {
        2:[0., 0.]
    }

T = boundaryconditions_2dof.neumann(mesh, traction)

B = R #+ T

K, B = boundaryconditions_2dof.dirichlet(K, B, mesh, displacement)

K = sparse.csc_matrix(K)

a = spsolve(K, B)

plotter.nodes_network_deformedshape2(mesh, a)
#plotter.nodes_network_edges(mesh)



plt.show()
