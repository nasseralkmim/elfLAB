__author__ = 'Nasser'
import numpy as np


def globalMatrix(K_ele, mesh):
    K = np.zeros((2*mesh.num_nodes, 2*mesh.num_nodes))

    for e in range(mesh.num_ele):
        for i in range(len(mesh.ele_conn[e])):
            for j in range(len(mesh.ele_conn[e])):
                # Dof 1-x
                K[2*mesh.ele_conn[e, i],
                  2*mesh.ele_conn[e, j]] += K_ele[2*i, 2*j, e]
                # Dof 1-y
                K[2*mesh.ele_conn[e, i],
                  2*mesh.ele_conn[e, j]+1] += K_ele[2*i, 2*j+1, e]

                K[2*mesh.ele_conn[e, i]+1,
                  2*mesh.ele_conn[e, j]] += K_ele[2*i+1, 2*j, e]

                K[2*mesh.ele_conn[e, i]+1,
                  2*mesh.ele_conn[e, j]+1] += K_ele[2*i+1, 2*j+1, e]

    return K


def globalVector(V_ele, mesh):
    V = np.zeros(2*mesh.num_nodes)

    for e in range(mesh.num_ele):
        for i in range(len(mesh.ele_conn[e])):
            V[2*mesh.ele_conn[e, i]] += V_ele[2*i, e]
            V[2*mesh.ele_conn[e, i]+1] += V_ele[2*i+1, e]
    return V
