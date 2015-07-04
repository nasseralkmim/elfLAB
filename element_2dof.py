__author__ = 'Nasser'

import numpy as np
import math


class Matrices:
    """Build the elemental matrices.

    Creates an object that has as attributes the elemental matrices.

    """
    def __init__(self, objectmesh):
        self.mesh = objectmesh

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
        C = np.zeros((3,3))
        C[0, 0] = 1.0
        C[1, 1] = 1.0
        C[1, 0] = nu
        C[0, 1] = nu
        C[2, 2] = (1.0 - nu)
        self.C = (E/(1.0-nu**2.0))*C

        self.gp = self.mesh.chi / math.sqrt(3)

        self.K = np.zeros((8, 8, self.mesh.num_ele))


        for e in range(self.mesh.num_ele):
            for w in range(4):
                self.mesh.basisFunction2D(self.gp[w])
                self.mesh.eleJacobian(self.mesh.nodes_coord[
                    self.mesh.ele_conn[e, :]])

                D = np.array([
                    [self.mesh.dphi_xi[0, 0], 0, self.mesh.dphi_xi[0, 1], 0,
                    self.mesh.dphi_xi[0, 2], 0, self.mesh.dphi_xi[0, 3], 0]
                             ,
                    [0, self.mesh.dphi_xi[1, 0], 0, self.mesh.dphi_xi[1, 1], 0,
                    self.mesh.dphi_xi[1, 2], 0, self.mesh.dphi_xi[1, 3]]
                             ,
                    [self.mesh.dphi_xi[1, 0], self.mesh.dphi_xi[0, 0],
                    self.mesh.dphi_xi[1, 1], self.mesh.dphi_xi[0, 1],
                    self.mesh.dphi_xi[1, 2], self.mesh.dphi_xi[0, 2],
                    self.mesh.dphi_xi[1, 3], self.mesh.dphi_xi[0, 3]]])

                self.K[:, :, e] += (np.dot(np.dot(np.transpose(D), C), D) *
                                    self.mesh.detJac)


    def load(self, q):
        """Build the load vector for the internal distributed load

        Args:
            mesh: object that includes the mesh atributes and methods for basis
                functions and transformation jacobians
            q (array of functions): internal distributed load

        """
        self.R = np.zeros((8, self.mesh.num_ele))

        for e in range(self.mesh.num_ele):
            for w in range(4):
                self.mesh.basisFunction2D(self.gp[w])
                self.mesh.eleJacobian(self.mesh.nodes_coord[
                    self.mesh.ele_conn[e]])

                x1_o_e1e2, x2_o_e1e2 = self.mesh.mapping(e)

                load = q(x1_o_e1e2, x2_o_e1e2)

                self.R[0, e] += load[0]*self.mesh.phi[0]*self.mesh.detJac
                self.R[1, e] += load[1]*self.mesh.phi[0]*self.mesh.detJac
                self.R[2, e] += load[0]*self.mesh.phi[1]*self.mesh.detJac
                self.R[3, e] += load[1]*self.mesh.phi[1]*self.mesh.detJac
                self.R[4, e] += load[0]*self.mesh.phi[2]*self.mesh.detJac
                self.R[5, e] += load[1]*self.mesh.phi[2]*self.mesh.detJac
                self.R[6, e] += load[0]*self.mesh.phi[3]*self.mesh.detJac
                self.R[7, e] += load[1]*self.mesh.phi[3]*self.mesh.detJac
