__author__ = 'Nasser'
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
import matplotlib.pyplot as plt
import gmsh
import element1dof
import assemble1dof
import plotter
import boundaryconditions1dof

def solver(meshName, material, internal_heat, flux_imposed,
           temperature_imposed, plotUndeformed, plotTemperature):

    mesh = gmsh.parse(meshName)

    ele = element1dof.Matrices(mesh)

    s = mesh.surfaces

    if len(material) != len(s):
        print('A surface material was not assigned! - Check the material '
              'Dictionary')

    surfaces_keys = []
    for surf in material.keys():
        surfaces_keys.append(surf)

    kDic = {}
    for i in range(len(s)):
        kDic[s[i]] = material[surfaces_keys[i]]

    def diffusivity(x1, x2):
        return kDic

    ele.stiffness(diffusivity)

    ele.load(internal_heat)

    K = assemble1dof.globalMatrix(ele.K, mesh)
    P0q = assemble1dof.globalVector(ele.R, mesh)

    P0t = boundaryconditions1dof.neumann(mesh, flux_imposed)

    P0 = P0q + P0t

    K, P0 = boundaryconditions1dof.dirichlet(K, P0, mesh, temperature_imposed)

    Ks = sparse.csc_matrix(K)

    U = spsolve(Ks, P0)

    dpi=90

    if plotTemperature['Contour'] == True:
        plotter.tricontourf(U, mesh, 'Temperature', 'hot'
                                                    ''
                                                    '', dpi)

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

    #plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')


    plt.show()
