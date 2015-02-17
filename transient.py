__author__ = 'Nasser'
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import gmsh
import element
import assemble
import plotter
import boundaryconditions


def transient(mesh, delta_t, a0):
    """Solves for each step.

    """
    ele = element.Matrices(mesh)

    s = mesh.surfaces
    def k(x1, x2):
        return {s[0]: 1}

    ele.stiffness(k)

    ele.mass()

    def load(x1, x2):
        return 0.0

    ele.load_internal(load)

    K = assemble.globalMatrix(ele.K, mesh)
    M = assemble.globalMatrix(ele.M, mesh)
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

    G = delta_t*(B - np.dot(K, a0)) + np.dot(M, a0)

    M = sparse.csc_matrix(M)

    a = spsolve(M, G)

    #plotter.trisurface(a, mesh)
    plotter.tricontourtransient(a, mesh)
    #plotter.nodes_network2(mesh)

    plt.show(block=False)
    return a


mesh = gmsh.parse('mesh4')

delta_t = 0.1

time_interval = 100

a0 = np.zeros((mesh.num_nodes, 1)) + 1

for i in range(time_interval):
    a = transient(mesh, delta_t, a0)
    a0 = np.reshape(a, (mesh.num_nodes, 1))

plt.show()

