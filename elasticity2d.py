__author__ = 'Nasser'

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
import matplotlib.pyplot as plt
import gmsh
import element2dof
import assemble2dof
import plotter
import boundaryconditions2dof
import processing

def solver(meshname, mat, body_forces, traction_imposed, displacement_imposed,
           plotDeformed, plotStress, plotUndeformed):


    mesh = gmsh.parse(meshname)


    ele = element2dof.Matrices(mesh)

    s = mesh.surfaces

    ele.stiffness(mat['nu'], mat['E'])


    ele.body_forces(body_forces)

    K = assemble2dof.globalMatrix(ele.K, mesh)
    P0q = assemble2dof.globalVector(ele.P0q, mesh)
    

    P0t = boundaryconditions2dof.neumann(mesh, traction_imposed)

    P0 = P0q + P0t

    K, P0 = boundaryconditions2dof.dirichlet(K, P0, mesh, displacement_imposed)

    Ks = sparse.csc_matrix(K)

    U = spsolve(Ks, P0)

    ele.nodal_forces(U)
    Pnode = assemble2dof.globalVector(ele.pEle, mesh)

    sNode, sEle, eEle = processing.stress_recovery_simple(mesh, U, ele)

    principal_max = processing.principal_stress_max(sNode[0], sNode[1], sNode[2])
    principal_min= processing.principal_stress_min(sNode[0], sNode[1], sNode[2])

    dpi = 90
    magf = plotDeformed['DeformationMagf']

    #PLOTTER CONTOUR MAPS
    if plotStress['s11'] == True:
        plotter.tricontourf(sNode[0]/10**3, mesh,
                                 'Stress 11 (kPa)','spring', dpi)

    if plotStress['s22'] == True:
        plotter.tricontourf(sNode[1]/10**3, mesh,
                                 'Stress 22 (kPa)','cool', dpi)

    if plotStress['s12'] == True:
        plotter.tricontourf(sNode[2]/10**3, mesh,
                                 'Stress 12 (kPa)','hsv', dpi)

    if plotStress['sPmax'] == True:
        plotter.tricontourf(principal_max/10**3, mesh,
                                 'Stress Principal Max (kPa)','autumn', dpi)

    if plotStress['sPmin'] == True:
        plotter.tricontourf(principal_min/10**3, mesh,
                                 'Stress Principal min (kPa)','winter', dpi)

    #PLOTTER DRAW UNDEFORMED SHAPE, ELEMENTS, LABELS, BC
    if plotUndeformed['Domain'] == True:
        plotter.draw_domain(mesh, 'Case Study', dpi, 'k')

    if plotUndeformed['Elements'] == True:
        plotter.draw_elements(mesh, 'Case Study', dpi, 'k')

    if plotUndeformed['ElementLabel'] == True:
        plotter.draw_elements_label(mesh, 'Case Study',dpi)

    if plotUndeformed['EdgesLabel'] == True:
        plotter.draw_edges_label(mesh, 'Case Study',dpi)

    if plotUndeformed['NodeLabel'] == True:
        plotter.draw_nodes_label(mesh, 'Case Study',dpi)

        #PLOTTER DEFORMED SHAPE
    if plotDeformed['DomainUndeformed'] == True:
        plotter.draw_domain(mesh, 'Deformed Shape', dpi, 'SteelBlue')

    if plotDeformed['ElementsUndeformed'] == True:
        plotter.draw_elements(mesh, 'Deformed Shape', dpi, 'SteelBlue')

    if plotDeformed['DomainDeformed'] == True:
        plotter.draw_deformed_domain(mesh, U, 'Deformed Shape', dpi, magf, 'Tomato')

    if plotDeformed['ElementsDeformed'] == True:
        plotter.draw_deformed_elements(mesh, U, 'Deformed Shape', dpi, magf,
                                       'Tomato', 1)

    plt.show()
