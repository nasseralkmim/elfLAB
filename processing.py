__author__ = 'Nasser'
import numpy as np
import assemble1dof
import assemble2dof

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

    stress_11 = assemble1dof.globalVectorAverage2(stress_11_ele, mesh)
    stress_22 = assemble1dof.globalVectorAverage2(stress_22_ele, mesh)
    stress_12 = assemble1dof.globalVectorAverage2(stress_12_ele, mesh)

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
        sp_max[i] = (s11[i]+s22[i])/2.0 + np.sqrt((s11[i] - s22[i])**2.0/2.0 +
                                               s12[i]**2.0)

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

def stress_recovery_simple(mesh, U, material):
    """

    :param mesh:
    :param d:
    :param C:
    :return:
    """
    stress_ele = np.zeros((3, 4, mesh.num_ele))
    strain_ele = np.zeros((3, 4, mesh.num_ele))

    for surf, mat in material.items():
        E = mat[0]
        nu = mat[1]

        C = np.zeros((3,3))

        C[0, 0] = 1.0
        C[1, 1] = 1.0
        C[1, 0] = nu
        C[0, 1] = nu
        C[2, 2] = (1.0 - nu)/2.0
        C = (E/(1.0-nu**2.0))*C

        for e in range(mesh.num_ele):
            if surf == mesh.ele_surface[e, 1]:
                for w in range(4):
                    # Evaluate basis function at nodes natural coordinates chi
                    mesh.basisFunction2D(mesh.chi[w])
                    mesh.eleJacobian(mesh.nodes_coord[
                        mesh.ele_conn[e, :]])

                    B = np.array([
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

                    uEle = np.array([
                        U[2*mesh.ele_conn[e, 0]],
                        U[2*mesh.ele_conn[e, 0]+1],
                        U[2*mesh.ele_conn[e, 1]],
                        U[2*mesh.ele_conn[e, 1]+1],
                        U[2*mesh.ele_conn[e, 2]],
                        U[2*mesh.ele_conn[e, 2]+1],
                        U[2*mesh.ele_conn[e, 3]],
                        U[2*mesh.ele_conn[e, 3]+1],
                    ])

                    eEle = np.dot(B, uEle)

                    strain_ele[:, w, e] = eEle

                    # w represents each node
                    stress_ele[:, w, e] = np.dot(C, eEle)



    # Nodal stress Averaged by number of elements sharing a node
    s11node = assemble1dof.globalVectorAverage2(stress_ele[0,:,:], mesh)
    s22node = assemble1dof.globalVectorAverage2(stress_ele[1,:,:], mesh)
    s12node = assemble1dof.globalVectorAverage2(stress_ele[2,:,:], mesh)

    s11node = np.reshape(s11node, len(s11node))
    s22node = np.reshape(s22node, len(s22node))
    s12node = np.reshape(s12node, len(s12node))

    sEle = np.array([element_stress(s11node, mesh),
            element_stress(s22node, mesh),
            element_stress(s12node, mesh)])


    e11node = assemble1dof.globalVectorAverage2(strain_ele[0,:,:], mesh)
    e22node = assemble1dof.globalVectorAverage2(strain_ele[1,:,:], mesh)
    e12node = assemble1dof.globalVectorAverage2(strain_ele[2,:,:], mesh)

    e11node = np.reshape(e11node, len(s11node))
    e22node = np.reshape(e22node, len(s22node))
    e12node = np.reshape(e12node, len(s12node))

    eEle = np.array([element_stress(e11node, mesh),
            element_stress(e22node, mesh),
            element_stress(e12node, mesh)])


    return [s11node, s22node, s12node], sEle, eEle



def element_stress(s, mesh):
    """

    :param s:
    :param mesh:
    :return:
    """
    s_ele = np.zeros(mesh.num_ele)

    for e in range(mesh.num_ele):
        s_e = np.array([
                    s[mesh.ele_conn[e, 0]],
                    s[mesh.ele_conn[e, 1]],
                    s[mesh.ele_conn[e, 2]],
                    s[mesh.ele_conn[e, 3]],
                ])

        s_ele[e] = (s_e[0] + s_e[1] + s_e[2] + s_e[3])/4.0

    return s_ele



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


def print_dof_values(v, mesh, name):
    """

    :param v:
    :param mesh:
    :return:
    """
    dof1 = v[::2]
    dof2 = v[1::2]

    dof1 = np.reshape(dof1, len(mesh.nodes_coord))
    dof2 = np.reshape(dof2, len(mesh.nodes_coord))
    print(name)

    print('{:<5}  {:^12}  {:^12}'.format('Node', 'dof1', 'dof2'))
    for i in range(len(mesh.nodes_coord)):
        print('{:<5d}  {: >12.5e}  {: >12.5e}'.format(i, dof1[i], dof2[i]))


def print_ele_values(v, mesh, name):
    """

    :param v:
    :param mesh:
    :param name:
    :return:
    """
    print(name)
    print('{:<5} {:^12} {:^12} {:^12} '.format('Ele', 'xx', 'yy', 'xy'))
    for i in range(mesh.num_ele):
        print('{:<5d}  {: >12.5f} {: >12.5f} {: >12.5f}'
              ''.format(i, v[0, i],v[1, i], v[2, i]))


def nodal_forces(K, U, P0, displacement):
    """

    :param K:
    :param U:
    :param P0:
    :param displacements:
    :return:
    """
    R = np.dot(K, U)

    for l in displacement(1, 1).keys():
        if l[0] == 'nodes' or l[0] == 'node':
            for n in l[1:]:
                rx = displacement(1,1)[l][0]
                ry = displacement(1,1)[l][1]

                if rx != 'free':
                    R[2*n] = R[2*n] - P0[2*n]

                if ry != 'free':
                    R[2*n+1] = R[2*n+1] - P0[2*n+1]

    return R

def sub_matrices(K, P, displacement):

    ie = []
    for l in displacement(1, 1).keys():
        if l[0] == 'nodes' or l[0] == 'node':
            for n in l[1:]:
                rx = displacement(1,1)[l][0]
                ry = displacement(1,1)[l][1]


                if rx != 'free':
                   ie.append([n, 0, rx])

                if ry != 'free':
                    ie.append([n, 1, ry])

    ide = []
    de = np.zeros(len(ie))
    for i in range(len(de)):
        de[i] = ie[i][2]

        if ie[i][1] == 0:
            ide.append(2*ie[i][0])
        if ie[i][1] == 1:
            ide.append(2*ie[i][0]+1)

    idf = []
    for i in range(len(P)):
        if i in ide:
            continue
        else:
            idf.append(i)


    Ke = np.zeros((len(ie), len(ie)))
    Ke = K[np.ix_(ide, ide)]
    Kf = K[np.ix_(idf, idf)]
    Kef = K[np.ix_(ide, idf)]
    Kfe = K[np.ix_(idf, ide)]
    Pf = P[np.ix_(idf)]
    Pe = P[np.ix_(ide)]

    return Ke, Kf, Kef, Kfe, de, Pf, Pe, ide, idf


def join_dof_values(de, df, ide, idf):

    U = np.zeros(len(ide)+len(idf))
    for i in range(len(ide)):
        U[ide[i]] = de[i]

    for i in range(len(idf)):
        U[idf[i]] = df[i]

    return U