__author__ = 'Nasser'

import numpy as np
import math


class Matrices:
    """Build the elemental matrices.

    Creates an object that has as attributes the elemental matrices.

    """
    def __init__(self, objectmesh):
        self.mesh = objectmesh
        mesh = self.mesh




    def stiffness(self, nu, E):
        """Build the elemental stiffness matrix.

        Runs over each individual element properties when the object methods
        are called. The object is the mesh and its methods are basisFunction2D
        which defines the phi function and its derivative;  eleJacobian which
        produce a Jacobian matrix and its determinant and also the derivative
        of the basis function with respect the spatial coordinates.

        .. note::

            How it works:

            1. Loop over elements.

            2. loop over the 4 possible combination of 2 GP for each direction.

            3. Call the methods to create the derivative of shape functions  and Jacobian for each GP combination.

            4. Build the stiffness matrix from a matrix multiplication.

        Gauss points from natural nodal coordinates::

             gp = [ [-1, -1]
                    [ 1, -1]
                    [ 1,  1]
                    [-1,  1] ]/ sqrt(3)


        Args:
            mesh: object that includes the mesh atributes and methods for basis
                functions and transformation jacobians.
            k (function) : material properties for the constitutive relation.

        Return:
            K_ele (array of float): 3nd order array 4x4 with the elemental
            stiffness matrix.

        """
        mesh = self.mesh
        C = np.zeros((3,3))
        C[0, 0] = 1.0
        C[1, 1] = 1.0
        C[1, 0] = nu
        C[0, 1] = nu
        C[2, 2] = (1.0 - nu)/2.0

        self.C = (E/(1.0-nu**2.0))*C

        self.gp = mesh.chi / math.sqrt(3.0)

        self.K = np.zeros((8, 8, mesh.num_ele))


        for e in range(mesh.num_ele):
            for w in range(4):
                mesh.basisFunction2D(self.gp[w])
                mesh.eleJacobian(mesh.nodes_coord[
                    mesh.ele_conn[e, :]])

                if mesh.detJac <= 0.0:
                    print('Error: non-positive Jacobian - '
                          'check element nodes numbering')
                    print('Element', e)

                dp_xi = mesh.dphi_xi
                B = np.array([
                    [dp_xi[0, 0], 0, dp_xi[0, 1], 0, dp_xi[0, 2], 0,
                     dp_xi[0, 3], 0]
                             ,
                    [0, dp_xi[1, 0], 0, dp_xi[1, 1], 0, dp_xi[1, 2], 0,
                     dp_xi[1, 3]]
                             ,
                    [dp_xi[1, 0], dp_xi[0, 0], dp_xi[1, 1], dp_xi[0, 1],
                    dp_xi[1, 2], dp_xi[0, 2],dp_xi[1, 3], dp_xi[0, 3]]])


                CB = np.dot(self.C, B)
                self.K[:, :, e] += (np.dot(np.transpose(B), CB) * mesh.detJac)




    def body_forces(self, q):
        """Build the load vector for the internal distributed load

        Args:
            mesh: object that includes the mesh attributes and methods for basis
                functions and transformation jacobians
            q (array of functions): internal distributed load

        """
        mesh = self.mesh
        self.P0q = np.zeros((8, mesh.num_ele))

        for e in range(mesh.num_ele):
            for w in range(4):
                mesh.basisFunction2D(self.gp[w])
                mesh.eleJacobian(mesh.nodes_coord[
                    mesh.ele_conn[e]])

                x1_o_e1e2, x2_o_e1e2 = mesh.mapping(e)

                load = q(x1_o_e1e2, x2_o_e1e2)

                self.P0q[0, e] += load[0]*mesh.phi[0]*mesh.detJac
                self.P0q[1, e] += load[1]*mesh.phi[0]*mesh.detJac
                self.P0q[2, e] += load[0]*mesh.phi[1]*mesh.detJac
                self.P0q[3, e] += load[1]*mesh.phi[1]*mesh.detJac
                self.P0q[4, e] += load[0]*mesh.phi[2]*mesh.detJac
                self.P0q[5, e] += load[1]*mesh.phi[2]*mesh.detJac
                self.P0q[6, e] += load[0]*mesh.phi[3]*mesh.detJac
                self.P0q[7, e] += load[1]*mesh.phi[3]*mesh.detJac


    def strain_initial(self, U0):
        """

        :param nu:
        :param E:
        :return:
        """
        mesh = self.mesh

        self.P0e = np.zeros((8, mesh.num_ele))

        self.u0 = np.zeros((8, mesh.num_ele))

        self.e0 = np.zeros((3, mesh.num_ele))

        # create u0 for each element based on nodal displacement
        for key in U0(1, 1).keys():
            if key[0] == 'nodes' or key[0] == 'node':
                for n in key[1:]:
                    for e in range(mesh.num_ele):
                        if n in mesh.ele_conn[e]:

                            # find node index within element
                            i = np.where(mesh.ele_conn[e]==n)[0][0]

                            x = mesh.nodes_coord[mesh.ele_conn[e, i], 0]
                            y = mesh.nodes_coord[mesh.ele_conn[e, i], 1]

                            u = U0(x, y)

                            self.u0[2*i, e] = u[key][0]
                            self.u0[2*i+1, e] = u[key][1]

        #compute element node force due initial displacement
        for e in range(mesh.num_ele):
            for w in range(4):
                mesh.basisFunction2D(self.gp[w])
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

                e0 = np.dot(B, self.u0[:, e])

                self.e0[:, e] += e0

                s0 = np.dot(self.C, e0)

                self.P0e[:, e] += np.dot(np.transpose(B), s0)*mesh.detJac


    def stress_initial(self, stress0):
        """

        :param nu:
        :param E:
        :return:
        """
        mesh = self.mesh
        self.P0s = np.zeros((8, mesh.num_ele))

        for e in range(mesh.num_ele):
            for w in range(4):
                mesh.basisFunction2D(self.gp[w])
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

                self.P0s[:, e] += np.dot(np.transpose(D), stress0)*mesh.detJac



    def nodal_forces(self, U):
        """

        :param U:
        :return:
        """
        mesh = self.mesh
        self.pEle = np.zeros((8, mesh.num_ele))
        for e in range(mesh.num_ele):
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

            self.pEle[:, e] = (np.dot(self.K[:, :, e], uEle) -
                               self.P0q[:, e])



    def traction(self, traction):
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
        mesh = self.mesh
        self.P0t = np.zeros((8, mesh.num_ele))


        gp = np.array([[[-1.0/math.sqrt(3), -1.0],
                        [1.0/math.sqrt(3), -1.0]],
                       [[1.0, -1.0/math.sqrt(3)],
                        [1.0, 1.0/math.sqrt(3)]],
                       [[-1.0/math.sqrt(3), 1.0],
                        [1.0/math.sqrt(3), 1.0]],
                       [[-1.0, -1.0/math.sqrt(3)],
                        [-1.0, 1.0/math.sqrt(3)]]])

        # compute traction for all nodes
        for l in traction(1,1).keys():
            if l[0] == 'line':
                for e, side, line in mesh.boundary_elements:
                    if line == l[1]:
                        for w in range(2):
                            mesh.basisFunction2D(gp[side, w])
                            mesh.eleJacobian(mesh.nodes_coord[mesh.ele_conn[
                                                               e, :]])

                            x1_o_e1e2, x2_o_e1e2 = mesh.mapping(e)

                            t = traction(x1_o_e1e2, x2_o_e1e2)

                            dL = mesh.ArchLength[side]

                            self.P0t[0, e] += mesh.phi[0]*t[l][0]*dL
                            self.P0t[1, e] += mesh.phi[0]*t[l][1]*dL
                            self.P0t[2, e] += mesh.phi[1]*t[l][0]*dL
                            self.P0t[3, e] += mesh.phi[1]*t[l][1]*dL
                            self.P0t[4, e] += mesh.phi[2]*t[l][0]*dL
                            self.P0t[5, e] += mesh.phi[2]*t[l][1]*dL
                            self.P0t[6, e] += mesh.phi[3]*t[l][0]*dL
                            self.P0t[7, e] += mesh.phi[3]*t[l][1]*dL



            if l[0] == 'nodes' or l[0] == 'node':
                for n in l[1:]:
                    for e in range(mesh.num_ele):
                        if n in mesh.ele_conn[e]:
                            t = traction(1, 1)
                            i = np.where(mesh.ele_conn[e]==n)[0][0]
                            self.P0t[2*i, e] = t[l][0]
                            self.P0t[2*i+1, e] = t[l][1]
