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
import processing

mesh = gmsh.parse('meshteste')

ele = element_2dof.Matrices(mesh)

s = mesh.surfaces

ele.stiffness(nu=0.1, E=10000.0)

def distributed_load(x1, x2):
    return np.array([
        5.0*x1**2,
        0.
    ])

ele.load(distributed_load)

K = assemble_2dof.globalMatrix(ele.K, mesh)
R = assemble_2dof.globalVector(ele.R, mesh)


def traction(x1, x2):
    return {}


def displacement(x1, x2):
    return {
        0:[0., 0.]
    }

T = boundaryconditions_2dof.neumann(mesh, traction)

B = R #+ T

K, B = boundaryconditions_2dof.dirichlet(K, B, mesh, displacement)

K = sparse.csc_matrix(K)

d = spsolve(K, B)

s11, s22, s12 = processing.stress_recovery(mesh, d, ele.C)

s11 = np.reshape(s11, len(s11))
s22 = np.reshape(s22, len(s22))
s12 = np.reshape(s12, len(s12))

plotter.tricontour1(s11, mesh)
plotter.tricontour2(s22, mesh)
plotter.tricontour3(s12, mesh)

plotter.nodes_network_deformedshape_contour(mesh, d)
plotter.nodes_network_deformedshape(mesh, d)
plotter.nodes_network_edges(mesh)




plt.show()
