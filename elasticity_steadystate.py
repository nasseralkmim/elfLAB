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

mesh = gmsh.parse('meshteste2')

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
        0:[0., 0.],
        2:[0., 0.],
        1:[1.*(x2-0.1), 0.],
        3:[-1.*(x2-0.1), 0.]
    }


def displacement(x1, x2):
    return {
    }

T = boundaryconditions_2dof.neumann(mesh, traction)

B = R + T

K, B = boundaryconditions_2dof.dirichlet(K, B, mesh, displacement)

K = sparse.csc_matrix(K)

d = spsolve(K, B)

s11, s22, s12 = processing.stress_recovery_simple(mesh, d, ele.C)
s11g, s22g, s12g = processing.stress_recovery_gauss(mesh, d, ele.C)

principal_max = processing.principal_stress_max(s11, s22, s12)
principal_maxg = processing.principal_stress_max(s11g, s22g, s12g)


principal_min = processing.principal_stress_max(s11, s22, s12)
von_mises = processing.von_misse(s11, s22, s12)


dpi = 70

#plotter.tricontour(principal_max, mesh, name='Gauss x Regular',
# label='Regular', color='r', dpi=dpi, ls='dotted')
#plotter.tricontour(principal_maxg, mesh, name='Gauss x Regular', label='Gauss',
#                   color='b', dpi=dpi, ls='dashed')

#plotter.tricontourf(von_mises, mesh, name='Von Mises', cmap='rainbow', dpi=dpi)

#plotter.tricontourf(principal_max, mesh, name='Principal Stress Max',
#                    cmap='autumn', dpi=dpi)
#plotter.tricontourf(principal_min, mesh, name='Principal Stress Min',
#                    cmap='cool', dpi=dpi)
plotter.tricontourf(s11g, mesh, name='Stress 11', cmap='spring', dpi=dpi)
#plotter.tricontourf(s22, mesh, name='Stress 22', cmap='summer', dpi=dpi)
#plotter.tricontourf(s12, mesh, name='Stress 12', cmap='winter', dpi=dpi)
#plotter.nodes_network_deformedshape_contour(mesh, d, pi)
#plotter.nodes_network_deformedshape(mesh, d, dpi)
#plotter.nodes_network_edges_label(mesh, dpi)
#plotter.nodes_network_edges_element_label(mesh, dpi)

plotter.nodes_network(mesh, dpi)
plotter.draw_elements_label(mesh, dpi)
plotter.draw_edges_label(mesh, dpi)
plotter.draw_nodes_label(mesh, dpi)

plotter.boundary_condition_dirichlet(displacement, mesh, dpi)
#plotter.boundary_condition_neumann_value(traction, mesh, dpi)
plotter.boundary_condition_neumann(traction, mesh, dpi)

#plt.axis('off')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
