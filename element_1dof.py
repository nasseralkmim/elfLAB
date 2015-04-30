import numpy as np
import math


class Matrices:
    """Build the elemental matrices.

    Creates an object that has as attributes the elemental matrices.

    """
    def __init__(self, objectmesh):
        self.mesh = objectmesh



    def stiffness(self, k):
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

        self.gp = self.mesh.chi / math.sqrt(3)

        B = np.zeros((2, 4))
        self.K = np.zeros((4, 4, self.mesh.num_ele))

        for surface in k(1,1).keys():
            for e in range(self.mesh.num_ele):
                if self.mesh.ele_surface[e, 1] == surface:
                    for w in range(4):
                        self.mesh.basisFunction2D(self.gp[w])
                        self.mesh.eleJacobian(self.mesh.nodes_coord[
                            self.mesh.ele_conn[e, :]])

                        x1_o_e1e2, x2_o_e1e2 = self.mesh.mapping(e)

                        kvar = k(x1_o_e1e2, x2_o_e1e2)[surface]

                        B = self.mesh.dphi_xi

                        self.K[:, :, e] += kvar*(np.dot(np.transpose(B), B) *
                                            self.mesh.detJac)


    def load(self, q):
        """Build the load vector thermal diffusivity.

        Args:
            mesh: object that includes the mesh atributes and methods for basis
                functions and transformation jacobians
            q: thermal diffusivity

        """
        self.R = np.zeros((4, self.mesh.num_ele))

        for e in range(self.mesh.num_ele):
            for w in range(4):
                self.mesh.basisFunction2D(self.gp[w])
                self.mesh.eleJacobian(self.mesh.nodes_coord[
                    self.mesh.ele_conn[e]])

                x1_o_e1e2, x2_o_e1e2 = self.mesh.mapping(e)

                load = q(x1_o_e1e2, x2_o_e1e2)

                self.R[0, e] += load*self.mesh.phi[0]*self.mesh.detJac
                self.R[1, e] += load*self.mesh.phi[1]*self.mesh.detJac
                self.R[2, e] += load*self.mesh.phi[2]*self.mesh.detJac
                self.R[3, e] += load*self.mesh.phi[3]*self.mesh.detJac



    def mass(self):
        """Build the mass matrix for each element.

        """
        self.M = np.zeros((4, 4, self.mesh.num_ele))

        for e in range(self.mesh.num_ele):
            for w in range(4):
                self.mesh.basisFunction2D(self.gp[w])
                self.mesh.eleJacobian(self.mesh.nodes_coord[
                    self.mesh.ele_conn[e]])

                self.M[0, :, e] += self.mesh.phi[0]*self.mesh.phi[:]
                self.M[1, :, e] += self.mesh.phi[1]*self.mesh.phi[:]
                self.M[2, :, e] += self.mesh.phi[2]*self.mesh.phi[:]
                self.M[3, :, e] += self.mesh.phi[3]*self.mesh.phi[:]

