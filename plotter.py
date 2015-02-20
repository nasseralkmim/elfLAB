import numpy as np
import matplotlib.mlab as ml
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mayavi import mlab
import networkx as nx
import scipy.interpolate
from scipy.spatial import cKDTree as KDTree


def trisurface(a, mesh):
    fig = plt.figure('Trisurface')
    ax = fig.add_subplot(111, projection='3d')

    c = mesh.nodes_coord

    X, Y, Z = c[:, 0], c[:, 1], a

    triangles = []
    for n1, n2, n3, n4 in mesh.ele_conn:
        triangles.append([n1, n2, n3])
        triangles.append([n1, n3, n4])

    triangles = np.asarray(triangles)

    Surf = ax.plot_trisurf(X, Y, triangles, Z, cmap='hot', linewidth=0.1)

    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$y$', fontsize=12)
    ax.set_zlabel(r'Temperature', fontsize=12)
    plt.colorbar(Surf)
    # plt.savefig('2.png', transparent=True)

    plt.draw()


def contour(a, nodes_coord, lev):
    c = nodes_coord

    X, Y, Z = c[:, 0], c[:, 1], a

    nx = 1000
    ny = 1000
    xi = np.linspace(min(X), max(X), nx)
    yi = np.linspace(min(Y), max(Y), ny)

    Xg, Yg = np.meshgrid(xi, yi)

    Zg = ml.griddata(X, Y, Z, Xg, Yg, interp='nn')

    CS2 = plt.contourf(Xg, Yg, Zg, lev, origin='lower', cmap='hot')
    plt.contour(Xg, Yg, Zg, lev, colors=('k', ), linewidth=(.7, ))
    plt.xlabel(r'$x$', fontsize=18)
    plt.ylabel(r'$y$', fontsize=18)
    cbar = plt.colorbar(CS2, shrink=0.8, extend='both')
    cbar.ax.set_ylabel('Temperature', fontsize=14)
    # plt.savefig('1.png', transparent=True, dpi=300)
    plt.show()


def contour3(a, nodes_coord, lev):
    c = nodes_coord

    X, Y, Z = c[:, 0], c[:, 1], a

    nx = len(nodes_coord[:,0])/2
    ny = len(nodes_coord[:,0])/2
    xi = np.linspace(min(X), max(X), nx)
    yi = np.linspace(min(Y), max(Y), ny)

    Xg, Yg = np.meshgrid(xi, yi)

    Zg = ml.griddata(X, Y, Z, Xg, Yg, interp='nn')

    CS2 = plt.contourf(Xg, Yg, Zg, lev, origin='lower', cmap='hot')
    plt.contour(Xg, Yg, Zg, lev, colors=('k', ), linewidth=(.7, ))
    plt.xlabel(r'$x$', fontsize=18)
    plt.ylabel(r'$y$', fontsize=18)
    cbar = plt.colorbar(CS2, shrink=0.8, extend='both')
    cbar.ax.set_ylabel('Temperature', fontsize=14)
    # plt.savefig('1.png', transparent=True, dpi=300)
    plt.show()


def nodes(nodes_coord):
    c = nodes_coord

    X, Y = c[:, 0], c[:, 1]

    plt.plot(X, Y, 'ok')
    plt.show()

def mayavi(a, mesh):
    c = mesh.nodes_coord

    Z = a

    X, Y = c[:, 0], c[:, 1]

    nx = 1000
    ny = 1000
    xi = np.linspace(min(X), max(X), nx)
    yi = np.linspace(min(Y), max(Y), ny)

    Xg, Yg = np.meshgrid(xi, yi)
    Zg = ml.griddata(X, Y, Z, Xg, Yg, interp='nn')

    mlab.imshow(Zg)
    mlab.show()

def nodes_network(mesh):

    c = mesh.nodes_coord

    X, Y = c[:, 0], c[:, 1]

    G = nx.Graph()

    label = []
    for i in range(len(X)):
        label.append(i)
        G.add_node(i, posxy=(X[i], Y[i]))

    if mesh.gmsh == 1.0:
        temp = np.copy(mesh.ele_conn[:, 2])
        mesh.ele_conn[:, 2] = mesh.ele_conn[:, 3]
        mesh.ele_conn[:, 3] = temp


    for i in range(len(mesh.ele_conn)):
        G.add_cycle([mesh.ele_conn[i, 0],
                     mesh.ele_conn[i, 1],
                     mesh.ele_conn[i, 3],
                     mesh.ele_conn[i, 2]], )


    edge_line_nodes = {}
    for i in range(len(mesh.boundary_nodes[:, 0])):
        edge_line_nodes[(mesh.boundary_nodes[i, 1], mesh.boundary_nodes[i,
                                                                        2])] \
            = mesh.boundary_nodes[i, 0]


    positions = nx.get_node_attributes(G, 'posxy')

    nx.draw_networkx(G, positions, node_size=5, node_color='k', font_size=0)

    limits=plt.axis('off')
    plt.show()


def nodes_network_edges(mesh):
    c = mesh.nodes_coord

    X, Y = c[:, 0], c[:, 1]
    plt.figure('Network')
    G = nx.Graph()

    label = []
    for i in range(len(X)):
        label.append(i)
        G.add_node(i, posxy=(X[i], Y[i]))

    if mesh.gmsh == 1.0:
        temp = np.copy(mesh.ele_conn[:, 2])
        mesh.ele_conn[:, 2] = mesh.ele_conn[:, 3]
        mesh.ele_conn[:, 3] = temp


    for i in range(len(mesh.ele_conn)):
        G.add_cycle([mesh.ele_conn[i, 0],
                     mesh.ele_conn[i, 1],
                     mesh.ele_conn[i, 3],
                     mesh.ele_conn[i, 2]], )


    edge_labels = {}
    for i in range(len(mesh.boundary_nodes[:, 0])):
        edge_labels[(mesh.boundary_nodes[i, 1], mesh.boundary_nodes[i,
                                                                        2])] \
            = str(mesh.boundary_nodes[i, 0])


    positions = nx.get_node_attributes(G, 'posxy')

    nx.draw_networkx(G, positions, node_size=1, node_color='k', font_size=0)
    nx.draw_networkx_edge_labels(G, positions, edge_labels, label_pos=0.5,
                                 font_size=7)

    limits=plt.axis('off')
    plt.draw()

def surface(a, nodes_coord):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    c = nodes_coord

    X, Y, Z = c[:, 0], c[:, 1], a

    nx = 200
    ny = 200
    xi = np.linspace(min(X), max(X), nx)
    yi = np.linspace(min(Y), max(Y), ny)

    Xg, Yg = np.meshgrid(xi, yi)

    Zg = ml.griddata(X, Y, Z, Xg, Yg, interp='linear')

    Surf = ax.plot_surface(Xg, Yg, Zg, cmap='hot', linewidth=0 )

    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$y$', fontsize=12)
    ax.set_zlabel(r'Temperature', fontsize=12)
    plt.colorbar(Surf)
    # plt.savefig('2.png', transparent=True)
    plt.show()

def contour2(a, mesh):

    c = mesh.nodes_coord
    bn = mesh.boundary_nodes

    xx, yy, zz = c[:, 0], c[:, 1], a

    ccx = np.append(c[bn[:, 1], 0], c[bn[0, 1], 0])
    ccy = np.append(c[bn[:, 1], 1], c[bn[0, 1], 1])

    triangles = []
    for n1, n2, n3, n4 in mesh.ele_conn:
        triangles.append([n1, n2, n3])
        triangles.append([n1, n3, n4])

    trianglesa = np.asarray(triangles)

    CS2 = plt.tricontourf(xx, yy, trianglesa, zz, 10, origin='lower',
                          cmap='hot')

    #plt.plot(ccx , ccy, '-k')
    #plt.scatter(xx, yy, c=zz)
    plt.xlabel(r'$x$', fontsize=18)
    plt.ylabel(r'$y$', fontsize=18)
    cbar = plt.colorbar(CS2, shrink=0.8, extend='both')
    cbar.ax.set_ylabel('Temperature', fontsize=14)

    X, Y = c[:, 0], c[:, 1]

    G2 = nx.Graph()

    label = []
    for i in range(len(X)):
        label.append(i)
        G2.add_node(i, posxy=(X[i], Y[i]))

    if mesh.gmsh == 1.0:
        temp = np.copy(mesh.ele_conn[:, 2])
        mesh.ele_conn[:, 2] = mesh.ele_conn[:, 3]
        mesh.ele_conn[:, 3] = temp


    for i in range(len(mesh.ele_conn)):
        G2.add_cycle([mesh.ele_conn[i, 0],
                     mesh.ele_conn[i, 1],
                     mesh.ele_conn[i, 3],
                     mesh.ele_conn[i, 2]], )


    edge_line_nodes = {}
    for i in range(len(mesh.boundary_nodes[:, 0])):
        edge_line_nodes[(mesh.boundary_nodes[i, 1], mesh.boundary_nodes[i,
                                                                        2])] \
            = mesh.boundary_nodes[i, 0]


    positions = nx.get_node_attributes(G2, 'posxy')
    nx.draw_networkx(G2, positions, node_size=1, node_color='k', font_size=0,
                     alpha=0.3)

    limits=plt.axis('off')
    # plt.savefig('1.png', transparent=True, dpi=300)
    plt.show()

def tricontour(a, mesh):
    """Plot contour with the tricoutour function and the boundary line with
    the boundary node.

    """
    plt.figure('Tricontour')
    c = mesh.nodes_coord
    bn = mesh.boundary_nodes

    xx, yy, zz = c[:, 0], c[:, 1], a

    ccx = np.append(c[bn[:, 1], 0], c[bn[0, 1], 0])
    ccy = np.append(c[bn[:, 1], 1], c[bn[0, 1], 1])

    triangles = []
    for n1, n2, n3, n4 in mesh.ele_conn:
        triangles.append([n1, n2, n3])
        triangles.append([n1, n3, n4])

    triangles = np.asarray(triangles)

    CS2 = plt.tricontourf(xx, yy, triangles, zz, 10, origin='lower',
                          cmap='hot')

    plt.plot(ccx , ccy, '-k')
    #plt.scatter(xx, yy, c=zz)
    plt.xlabel(r'$x$', fontsize=18)
    plt.ylabel(r'$y$', fontsize=18)
    cbar = plt.colorbar(CS2, shrink=0.8, extend='both')
    cbar.ax.set_ylabel('Temperature', fontsize=14)

    limits=plt.axis('off')
    # plt.savefig('1.png', transparent=True, dpi=300)
    plt.draw()

def tricontour_transient(a, mesh, i):
    """Plot contour with the tricoutour function and the boundary line with
    the boundary node.

    """
    #plt.figure('Tricontour - '+str(i))
    c = mesh.nodes_coord
    bn = mesh.boundary_nodes

    xx, yy, zz = c[:, 0], c[:, 1], a

    ccx = np.append(c[bn[:, 1], 0], c[bn[0, 1], 0])
    ccy = np.append(c[bn[:, 1], 1], c[bn[0, 1], 1])

    triangles = []
    for n1, n2, n3, n4 in mesh.ele_conn:
        triangles.append([n1, n2, n3])
        triangles.append([n1, n3, n4])

    triangles = np.asarray(triangles)

    CS2 = plt.tricontourf(xx, yy, triangles, zz, 10, origin='lower',
                          cmap='hot')

    #plt.plot(ccx , ccy, '-k')
    #plt.scatter(xx, yy, c=zz)
    #plt.xlabel(r'$x$', fontsize=18)
    #plt.ylabel(r'$y$', fontsize=18)
    #cbar = plt.colorbar(CS2, shrink=0.8, extend='both')
    #cbar.ax.set_ylabel('Temperature', fontsize=14)

    limits=plt.axis('off')
    #plt.savefig(str(i)+'.eps', transparent=True, dpi=300)
    plt.draw()

