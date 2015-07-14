__author__ = 'Nasser'
import numpy as np
import assemble_1dof


def stress_recovery_gauss(mesh, d, C):
    """

    :param mesh:
    :param d:
    :return:
    """
    O = np.array([
        [1.+np.sqrt(3.)/2., -.5, 1.-np.sqrt(3.)/2., -.5],
        [-.5, 1.+np.sqrt(3.)/2., -.5, 1.-np.sqrt(3.)/2.],
        [1.-np.sqrt(3.)/2., -.5, 1.+np.sqrt(3.)/2., -.5],
        [-.5, 1.-np.sqrt(3.)/2., -.5, 1.+np.sqrt(3.)/2.]
    ])
    stress_ele = np.zeros((3, 4, mesh.num_ele))
    stress_11_ele = np.zeros((4, mesh.num_ele))
    stress_22_ele = np.zeros((4, mesh.num_ele))
    stress_12_ele = np.zeros((4, mesh.num_ele))

    for e in range(mesh.num_ele):
        for w in range(4):
            mesh.basisFunction2D(mesh.chi[w]/np.sqrt(3.))
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
                d[2*mesh.ele_conn[e, 0]],
                d[2*mesh.ele_conn[e, 0]+1],
                d[2*mesh.ele_conn[e, 1]],
                d[2*mesh.ele_conn[e, 1]+1],
                d[2*mesh.ele_conn[e, 2]],
                d[2*mesh.ele_conn[e, 2]+1],
                d[2*mesh.ele_conn[e, 3]],
                d[2*mesh.ele_conn[e, 3]+1],
            ])

            strain = np.dot(D, d_ele)

            stress_ele[:, w, e] = np.dot(C, strain)

            stress_11_ele[:, e] = np.dot(O, stress_ele[0, :, e])
            stress_22_ele[:, e] = np.dot(O, stress_ele[1, :, e])
            stress_12_ele[:, e] = np.dot(O, stress_ele[2, :, e])

    stress_11 = assemble_1dof.globalVectorAverage2(stress_11_ele, mesh)
    stress_22 = assemble_1dof.globalVectorAverage2(stress_22_ele, mesh)
    stress_12 = assemble_1dof.globalVectorAverage2(stress_12_ele, mesh)

    stress_11 = np.reshape(stress_11, len(stress_11))
    stress_22 = np.reshape(stress_22, len(stress_22))
    stress_12 = np.reshape(stress_12, len(stress_12))

    return stress_11, stress_22, stress_12



def principal_stress_max(s11, s22, s12):
    """

    :param s11:
    :param s22:
    :param s12:
    :return:
    """
    sp_max = np.zeros(len(s11))
    for i in range(len(s11)):
        sp_max[i] = (s11[i]+s22[i])/2. + np.sqrt((s11[i] - s22[i])**2./2. +
                                               s12[i]**2.)

    return sp_max


def principal_stress_min(s11, s22, s12):
    """

    :param s11:
    :param s22:
    :param s12:
    :return:
    """
    sp_min = np.zeros(len(s11))
    for i in range(len(s11)):
        sp_min[i] = (s11[i]+s22[i])/2. - np.sqrt((s11[i] - s22[i])**2./2. +
                                               s12[i]**2.)

    return sp_min

def stress_recovery_simple(mesh, d, C):
    """

    :param mesh:
    :param d:
    :param C:
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
                d[2*mesh.ele_conn[e, 0]],
                d[2*mesh.ele_conn[e, 0]+1],
                d[2*mesh.ele_conn[e, 1]],
                d[2*mesh.ele_conn[e, 1]+1],
                d[2*mesh.ele_conn[e, 2]],
                d[2*mesh.ele_conn[e, 2]+1],
                d[2*mesh.ele_conn[e, 3]],
                d[2*mesh.ele_conn[e, 3]+1],
            ])

            strain = np.dot(D, d_ele)

            # w represents each node
            stress_ele[:, w, e] = np.dot(C, strain)

    stress_11 = assemble_1dof.globalVectorAverage2(stress_ele[0,:,:], mesh)
    stress_22 = assemble_1dof.globalVectorAverage2(stress_ele[1,:,:], mesh)
    stress_12 = assemble_1dof.globalVectorAverage2(stress_ele[2,:,:], mesh)

    stress_11 = np.reshape(stress_11, len(stress_11))
    stress_22 = np.reshape(stress_22, len(stress_22))
    stress_12 = np.reshape(stress_12, len(stress_12))

    return stress_11, stress_22, stress_12


def von_misse(s11, s22, s12):
    """

    :param s11:
    :param s22:
    :param s12:
    :return:
    """
    return  np.sqrt((s11/2. - s22/2.)**2. + (s22/2.)**2. + (-s11/2.)**2. +
                                             3.*s12**2.)

def von_misse2(s11, s22, s12):
    """

    :param s11:
    :param s22:
    :param s12:
    :return:
    """
    return  np.sqrt(s11**2. - s11*s22 + s22**2. + 3.*s12**2.)