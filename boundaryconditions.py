import numpy as np
import assemble
from scipy import sparse
import math
from numba import jit


def dirichlet(K, B, mesh, temperature):
    """Apply Dirichlet BC.

    .. note::

        How its done:

        1. Loop over the lines where Dirichlet boundary conditions are applied.

        2. loop over the lines at the boundary.

        3. If this nodes, which is identified as follows::

            boundary_nodes = [line node1 node2]

            are in the line where dirichlet BC were specified, change the
            stiffness matrix and the B vector based on this node index.

    Args:
        K: Stiffness matrix.
        B: Vector with the load and traction.
        temperature: Function with 4 components.
        temperature_sides: Boundary lines where dirichlet boundary conditions
            are applied.

    Returns:
        K: Modified stiffness matrix.

        B: Modified vector.

    """
    for line in temperature(1,1).keys():
        for n in range(len(mesh.boundary_nodes[:, 0])):
            if line == mesh.boundary_nodes[n, 0]:
                K[mesh.boundary_nodes[n, 1], :] = 0.0
                K[mesh.boundary_nodes[n, 2], :] = 0.0

                K[mesh.boundary_nodes[n, 1], mesh.boundary_nodes[n, 1]] =  1.0
                K[mesh.boundary_nodes[n, 2], mesh.boundary_nodes[n, 2]] = 1.0

                t1 = temperature(mesh.nodes_coord[mesh.boundary_nodes[n,1], 0],
                                mesh.nodes_coord[mesh.boundary_nodes[n, 1], 1],)

                t2 = temperature(mesh.nodes_coord[mesh.boundary_nodes[n, 2], 0],
                                mesh.nodes_coord[mesh.boundary_nodes[n, 2], 1],)

                B[mesh.boundary_nodes[n, 1]] = t1[line]
                B[mesh.boundary_nodes[n, 2]] = t2[line]

    K = sparse.csc_matrix(K)
    return K, B

@jit
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
    Tele = np.zeros((4, mesh.num_ele))


    gp = np.array([[[-1.0/math.sqrt(3), -1.0],
                    [1.0/math.sqrt(3), -1.0]],
                   [[1.0, -1.0/math.sqrt(3)],
                    [1.0, 1.0/math.sqrt(3)]],
                   [[-1.0/math.sqrt(3), 1.0],
                    [1.0/math.sqrt(3), 1.0]],
                   [[-1.0, -1.0/math.sqrt(3)],
                    [-1.0, 1/math.sqrt(3)]]])


    for line in traction(1,1).keys():
        for ele, side, l in mesh.boundary_elements:
            if l == line:
                for w in range(2):
                    mesh.basisFunction2D(gp[side, w])
                    mesh.eleJacobian(mesh.nodes_coord[mesh.ele_conn[ele, :]])

                    x1_o_e1e2, x2_o_e1e2 = mesh.mapping(ele)
                    t = traction(x1_o_e1e2, x2_o_e1e2)

                    Tele[0, ele] += mesh.phi[0]*t[l]*mesh.ArchLength[side]
                    Tele[1, ele] += mesh.phi[1]*t[l]*mesh.ArchLength[side]
                    Tele[2, ele] += mesh.phi[2]*t[l]*mesh.ArchLength[side]
                    Tele[3, ele] += mesh.phi[3]*t[l]*mesh.ArchLength[side]

    T = assemble.globalVector(Tele, mesh)

    return T