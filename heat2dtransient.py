__author__ = 'Nasser'
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import gmsh
import element1dof
import assemble1dof
import plotter
import boundaryconditions1dof


def transient(mesh, delta_t, a0, i):
    """Solves for each step.

    """
    ele = element1dof.Matrices(mesh)

    s = mesh.surfaces
    def k(x1, x2):
        return {s[0]: 1}

    ele.stiffness(k)

    ele.mass()

    def load(x1, x2):
        return 0.0

    ele.load(load)

    K = assemble1dof.globalMatrix(ele.K, mesh)
    M = assemble1dof.globalMatrix(ele.M, mesh)
    R = assemble1dof.globalVector(ele.R, mesh)

    def traction(x1, x2):
        return {}

    def temperature(x1, x2):
        return {0:20-x2,
                1:20-x2,
                2:15,
                3:27,
                4:27,
                5:27}

    T = boundaryconditions1dof.neumann(mesh, traction)

    B = R + T

    K, B = boundaryconditions1dof.dirichlet(K, B, mesh, temperature)

    G = delta_t*(B - np.dot(K, a0)) + np.dot(M, a0)

    M = sparse.csc_matrix(M)

    a = spsolve(M, G)

    #plotter.trisurface(a, mesh)
    plotter.tricontour_transient(a, mesh, i)
    #plotter.nodes_network_edges(mesh)

    plt.show(block=False)
    return a


mesh = gmsh.parse('mesh4')

delta_t = 0.15

time_interval = 5

a0 = np.zeros((mesh.num_nodes, 1))+2

for i in range(time_interval):
    a = transient(mesh, delta_t, a0, i)
    a0 = np.reshape(a, (mesh.num_nodes, 1))

plt.show()

