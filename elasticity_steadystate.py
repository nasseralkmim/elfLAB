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

mesh = gmsh.parse('meshvalidation')

ele = element_2dof.Matrices(mesh)

s = mesh.surfaces

ele.stiffness(nu=0.3, E=200*10**6)

def distributed_load(x1, x2):
    return np.array([
        0.,
        0.,
    ])

ele.load(distributed_load)

K = assemble_2dof.globalMatrix(ele.K, mesh)
R = assemble_2dof.globalVector(ele.R, mesh)


def traction(x1, x2):
    return {
        1:[0., 0.],
        2:[0., 0.],
        3:[0.1, 0.]
    }


def displacement(x1, x2):
    return {
        0:[0., 0.]
    }

T = boundaryconditions_2dof.neumann(mesh, traction)

B = R + T

K, B = boundaryconditions_2dof.dirichlet(K, B, mesh, displacement)

K = sparse.csc_matrix(K)

d = spsolve(K, B)

s11, s22, s12 = processing.stress_rec2(mesh, d, ele.C)

principal_11 = processing.principal_stress(s11, s22, s12)

von_mises = processing.von_misse(s11, s22, s12)


dpi = 100

print(mesh.num_ele)

#plotter.nodes_network_edges_element_label(mesh, dpi=dpi)
plotter.tricontour(von_mises, mesh, name='Von Mises', cmap='rainbow', dpi=dpi)
plotter.tricontour(principal_11, mesh, name='Principal Stress', cmap='cool',dpi=dpi)
plotter.tricontour(s11, mesh, name='Stress 11', cmap='spring', dpi=dpi)
#plotter.tricontour(s22, mesh, name='Stress 22', cmap='summer', dpi=dpi)
plotter.tricontour(s12, mesh, name='Stress 12', cmap='winter', dpi=dpi)
#plotter.nodes_network_deformedshape_contour(mesh, d, dpi=dpi)
plotter.nodes_network_deformedshape(mesh, d, dpi=dpi)
plotter.nodes_network_edges(mesh, dpi=dpi)


plt.show()
