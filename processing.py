__author__ = 'Nasser'
import numpy as np
import assemble_1dof


def stress_recovery(mesh, d, C):
    """

    :param mesh:
    :param d:
    :return:
    """
    stress_ele = np.zeros((3, 4, mesh.num_ele))
    for e in range(mesh.num_ele):
        for w in range(4):
            mesh.basisFunction2D(mesh.chi[w])
            mesh.eleJacobian(mesh.nodes_coord[
                mesh.ele_conn[e, :]])

            D = np.array([
                [mesh.dphi_xi[0, 0], 0, mesh.dphi_xi[0, 1], 0,
                mesh.dphi_xi[0, 2], 0, mesh.dphi_xi[0, 3], 0]
                         ,
                [0, mesh.dphi_xi[1, 0], 0, mesh.dphi_xi[1, 1], 0,
                mesh.dphi_xi[1, 2], 0, mesh.dphi_xi[1, 3]]
                         ,
                [mesh.dphi_xi[1, 0], mesh.dphi_xi[0, 0],
                mesh.dphi_xi[1, 1], mesh.dphi_xi[0, 1],
                mesh.dphi_xi[1, 2], mesh.dphi_xi[0, 2],
                mesh.dphi_xi[1, 3], mesh.dphi_xi[0, 3]]])

            d_ele = np.array([
                2*mesh.ele_conn[e, 0],
                2*mesh.ele_conn[e, 0]+1,
                2*mesh.ele_conn[e, 1],
                2*mesh.ele_conn[e, 1]+1,
                2*mesh.ele_conn[e, 2],
                2*mesh.ele_conn[e, 2]+1,
                2*mesh.ele_conn[e, 3],
                2*mesh.ele_conn[e, 3]+1,
            ])

            strain = np.dot(D, d_ele)

            stress_ele[:, w, e] = np.dot(C, strain)

    stress_11 = assemble_1dof.globalVector(stress_ele[0,:,:], mesh)
    stress_22 = assemble_1dof.globalVector(stress_ele[1,:,:], mesh)
    stress_12 = assemble_1dof.globalVector(stress_ele[2,:,:], mesh)

    return stress_11, stress_22, stress_12