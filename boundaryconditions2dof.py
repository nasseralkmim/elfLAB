__author__ = 'Nasser'
import numpy as np
import assemble2dof
from scipy import sparse
import math
from numba import jit


def dirichlet(K, F, mesh, displacement):
    """Apply Dirichlet BC.

    .. note::

        How its done:

        1. Loop over the lines where Dirichlet boundary conditions are
        applied. This is specified as key argument of the dictionary. The
        boundaries are defined on the gmsh nad tagged with a number.

        2. loop over all the nodes at the boundaries.

        3. If those nodes, identified as follows::

            boundary_nodes = [line node1 node2]

            are in the line where dirichlet BC were specified, change the
            stiffness matrix and the B vector based on this node index.

    The assignment can be done in two ways:

    1. with a dictionary specifying the boundary line:

        ['line', line]:[dx, dy]

    which means: the boundary line with tag line has a displacement dx on
    the x-direction and dy on the y-direction.

    2. with a dictionary with a specific node:

        ['node', node]:[0, 1]

    this way: the string 'node'identifies the type of boundary displacement
    that is been assigned; the number 0 identifies a free to move degree of
    freedom and 1 a restrained one. The first entry is the x-direction and
    the second y-direction.

    Args:
        K (2nd order array): Stiffness matrix.
        B (1st order array): Vector with the load and traction.
        temperature (function): Function with 4 components.

    Returns:
        K (2nd order array), B (1st order array): Modified stiffness matrix
        and vector.



    """
    for line in displacement(1,1).keys():
        if line[0] == 'line':
            for n in range(len(mesh.boundary_nodes[:, 0])):
                if line[1] == mesh.boundary_nodes[n, 0]:
                    K[2*mesh.boundary_nodes[n, 1], :] = 0.0
                    K[2*mesh.boundary_nodes[n, 1]+1, :] = 0.0
                    K[2*mesh.boundary_nodes[n, 2], :] = 0.0
                    K[2*mesh.boundary_nodes[n, 2]+1, :] = 0.0

                    K[2*mesh.boundary_nodes[n, 1], 2*mesh.boundary_nodes[n,1]] =  1.0
                    K[2*mesh.boundary_nodes[n, 1]+1,2* mesh.boundary_nodes[n,1]+1] =1.0
                    K[2*mesh.boundary_nodes[n, 2], 2*mesh.boundary_nodes[n,2]] = 1.0
                    K[2*mesh.boundary_nodes[n, 2]+1, 2*mesh.boundary_nodes[n,2]+1] =1.0

                    d1 = displacement(
                        mesh.nodes_coord[mesh.boundary_nodes[n, 1], 0],
                        mesh.nodes_coord[mesh.boundary_nodes[n, 1], 1])

                    d2 = displacement(
                        mesh.nodes_coord[mesh.boundary_nodes[n, 2], 0],
                        mesh.nodes_coord[mesh.boundary_nodes[n, 2], 1])

                    F[2*mesh.boundary_nodes[n, 1]] = d1[line][0]
                    F[2*mesh.boundary_nodes[n, 1]+1] = d1[line][1]
                    F[2*mesh.boundary_nodes[n, 2]] = d2[line][0]
                    F[2*mesh.boundary_nodes[n, 2]+1] = d2[line][1]

        if line[0] == 'nodes' or line[0] == 'node':

            rx = displacement(1, 1)[line][0]
            ry = displacement(1, 1)[line][1]

            for n in line[1:]:
                if rx != 'free':
                    K[2*n, :] = 0.0
                    K[2*n, 2*n] = 1.0
                    F[2*n] = rx

                if ry != 'free':
                    K[2*n + 1, :] = 0.0
                    K[2*n + 1, 2*n + 1] = 1.0
                    F[2*n + 1] = ry

        # modify Linear system matrix and vector for imposed 0.0 displacement
        if line[0] == 'support':

            for n in line[1:]:
                rx = displacement(1, 1)[line][0]
                ry = displacement(1, 1)[line][1]
                if rx != 'free':
                    K[2*n, :] = 0.0
                    K[2*n, 2*n] = 1.0
                    F[2*n] = 0.0

                if ry != 'free':
                    K[2*n + 1, :] = 0.0
                    K[2*n + 1, 2*n + 1] = 1.0
                    F[2*n + 1] = 0.0

    return K, F


def neumann(mesh, traction):
    """Apply Neumann BC.

    Computes the integral from the weak form with the boundary term.

    .. note::

        How its done:

        1. Define an array with the Gauss points for each path, each path will
        have 2 sets of two gauss points. One of the gp is fixed, which indicates
        that its going over an specific boundary of the element.

        Gauss points are defines as::

            gp = [gp, -1    1st path --> which has 2 possibilities for gp.
                  1,  gp    2nd path
                  gp,  1    3rd
                  -1,  gp]  4th

        2. A loop over the elements on the boundary extracting also the side
        where the boundary is located on this element.

    .. note::

        Edge elements are necessary because we need to extract the nodes
        from the connectivity. Then we can create a T for this element.

    Args:
        traction: Function with the traction and the line where the traction
            is applied.
        mesh: Object with the mesh attributes.

    Returns:
        T: Traction vector with size equals the dof.

    """
    Tele = np.zeros((8, mesh.num_ele))


    gp = np.array([[[-1.0/math.sqrt(3), -1.0],
                    [1.0/math.sqrt(3), -1.0]],
                   [[1.0, -1.0/math.sqrt(3)],
                    [1.0, 1.0/math.sqrt(3)]],
                   [[-1.0/math.sqrt(3), 1.0],
                    [1.0/math.sqrt(3), 1.0]],
                   [[-1.0, -1.0/math.sqrt(3)],
                    [-1.0, 1.0/math.sqrt(3)]]])


    for line in traction(1,1).keys():
        if line[0] == 'line':
            for ele, side, l in mesh.boundary_elements:
                if l == line[1]:
                    for w in range(2):
                        mesh.basisFunction2D(gp[side, w])
                        mesh.eleJacobian(mesh.nodes_coord[mesh.ele_conn[ele, :]])

                        x1_o_e1e2, x2_o_e1e2 = mesh.mapping(ele)

                        t = traction(x1_o_e1e2, x2_o_e1e2)

                        dL = mesh.ArchLength[side]
                        Tele[0, ele] += mesh.phi[0]*t[line][0]*dL
                        Tele[1, ele] += mesh.phi[0]*t[line][1]*dL
                        Tele[2, ele] += mesh.phi[1]*t[line][0]*dL
                        Tele[3, ele] += mesh.phi[1]*t[line][1]*dL
                        Tele[4, ele] += mesh.phi[2]*t[line][0]*dL
                        Tele[5, ele] += mesh.phi[2]*t[line][1]*dL
                        Tele[6, ele] += mesh.phi[3]*t[line][0]*dL
                        Tele[7, ele] += mesh.phi[3]*t[line][1]*dL


    T = assemble2dof.globalVector(Tele, mesh)

    for n in traction(1, 1).keys():
        if n[0] == 'node':
            t = traction(1,1)
            T[2*n[1]] = t[n][0]
            T[2*n[1]+1] = t[n][1]

    return T


def dirichlet2(mesh, displacement, K):
    """

    :param K:
    :param F:
    :param mesh:
    :param displacement:
    :return:
    """
    # build initial displacement vector
    U0 = np.zeros(mesh.num_nodes*2)

    for l in displacement(1, 1).keys():
        if l[0] == 'nodes' or l[0] == 'node':
            for n in l[1:]:
                u0 = displacement(1, 1)
                U0[2*n] = u0[l][0]
                U0[2*n+1] = u0[l][1]

    P0u = np.dot(K, U0)

    return P0u, U0


def dirichlet3(mesh, displacement):
    u0Ele = np.zeros((8, mesh.num_ele))


    for l in displacement(1, 1).keys():
        if l[0] == 'nodes' or l[0] == 'node':
            for n in l[1:]:
                for e in range(mesh.num_ele):
                    if n in mesh.ele_conn[e]:
                        u0 = displacement(1, 1)
                        i = np.where(mesh.ele_conn[e]==n)[0][0]
                        u0Ele[2*i, e] = u0[l][0]
                        u0Ele[2*i+1, e] = u0[l][1]

    return u0Ele