import numpy as np
import matplotlib.mlab as ml
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import networkx as nx
import scipy.interpolate
from scipy.spatial import cKDTree as KDTree


def trisurface(a, mesh, dpi):
    fig = plt.figure('Trisurface', dpi=dpi)
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


def nodes_network(mesh, dpi):

    c = mesh.nodes_coord

    X, Y = c[:, 0], c[:, 1]
    plt.figure('Elements', dpi=dpi)
    G2 = nx.Graph()

    label = []
    for i in range(len(X)):
        label.append(i)
        G2.add_node(i, posxy=(X[i], Y[i]))

    if mesh.gmsh == 1.0:
        temp = np.copy(mesh.ele_conn[:, 2])
        mesh.ele_conn[:, 2] = mesh.ele_conn[:, 3]
        mesh.ele_conn[:, 3] = temp
        mesh.gmsh += 1


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

    nx.draw_networkx(G2, positions, node_size=5, node_color='k', font_size=0)

    #limits=plt.axis('off')


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


def tricontourf(a, mesh, name, cmap, dpi):
    """Plot contour with the tricoutour function and the boundary line with
    the boundary node.

    """
    plt.figure(name, dpi=dpi)
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
                          cmap=cmap)

    plt.plot(ccx , ccy, '-k')
    #plt.scatter(xx, yy, c=zz)
    plt.xlabel(r'$x$', fontsize=18)
    plt.ylabel(r'$y$', fontsize=18)
    cbar = plt.colorbar(CS2, shrink=0.8, extend='both')
    cbar.ax.set_ylabel(name, fontsize=14)

    #limits=plt.axis('off')
    # plt.savefig('1.png', transparent=True, dpi=300)
    plt.axes().set_aspect('equal')

    plt.draw()


def tricontour(a, mesh, name, label, color, dpi, ls):
    """Plot contour with the tricoutour function and the boundary line with
    the boundary node.

    """
    plt.figure(name, dpi=dpi)
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

    CS2 = plt.tricontour(xx, yy, triangles, zz, 10, origin='lower',
                         linestyles=ls, colors=color)

    plt.plot(ccx , ccy, '-k')
    #plt.scatter(xx, yy, c=zz)
    plt.xlabel(r'$x$', fontsize=18)
    plt.ylabel(r'$y$', fontsize=18)
    cbar = plt.colorbar(CS2, shrink=0.8, extend='both')
    cbar.ax.set_ylabel(label, fontsize=14)


    #limits=plt.axis('off')
    # plt.savefig('1.png', transparent=True, dpi=300)
    plt.axes().set_aspect('equal')

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


def nodes_network_deformedshape(mesh, a, dpi):
    c = mesh.nodes_coord

    X, Y = c[:, 0], c[:, 1]
    dX, dY = c[:, 0] + a[::2], c[:, 1] + a[1::2]
    plt.figure('Deformation', dpi=dpi)

    G = nx.Graph()

    label = []
    for i in range(len(X)):
        label.append(i)
        G.add_node(i, posxy=(X[i], Y[i]))

    if mesh.gmsh == 1.0:
        temp = np.copy(mesh.ele_conn[:, 2])
        mesh.ele_conn[:, 2] = mesh.ele_conn[:, 3]
        mesh.ele_conn[:, 3] = temp
        mesh.gmsh += 1.0


    for i in range(len(mesh.ele_conn)):
        G.add_cycle([mesh.ele_conn[i, 0],
                     mesh.ele_conn[i, 1],
                     mesh.ele_conn[i, 3],
                     mesh.ele_conn[i, 2]], )

    positions = nx.get_node_attributes(G, 'posxy')

    nx.draw_networkx(G, positions, node_size=0, node_color='k', font_size=0,
                     style='dashed', width=0.5)
    G2 = nx.Graph()

    label2 = []
    for i in range(len(dX)):
        label2.append(i)
        G2.add_node(i, posxy2=(dX[i], dY[i]))

    for i in range(len(mesh.ele_conn)):
        G2.add_cycle([mesh.ele_conn[i, 0],
                     mesh.ele_conn[i, 1],
                     mesh.ele_conn[i, 3],
                     mesh.ele_conn[i, 2]], )

    positions2 = nx.get_node_attributes(G2, 'posxy2')

    nx.draw_networkx(G2, positions2, node_size=0, node_color='k',
                        font_size=0)

    plt.axes().set_aspect('equal')
    #limits=plt.axis('off')



def nodes_network_edges_label(mesh, dpi):
    c = mesh.nodes_coord

    X, Y = c[:, 0], c[:, 1]
    plt.figure('Elements', dpi=dpi)
    G = nx.Graph()

    label = []
    for i in range(len(X)):
        label.append(i)
        G.add_node(i, posxy=(X[i], Y[i]))

    if mesh.gmsh == 1.0:
        temp = np.copy(mesh.ele_conn[:, 2])
        mesh.ele_conn[:, 2] = mesh.ele_conn[:, 3]
        mesh.ele_conn[:, 3] = temp
        mesh.gmsh += 1

    for i in range(len(mesh.ele_conn)):
        G.add_cycle([mesh.ele_conn[i, 0],
                     mesh.ele_conn[i, 1],
                     mesh.ele_conn[i, 3],
                     mesh.ele_conn[i, 2]], )

    bound_middle = {}
    iant = mesh.boundary_nodes[0, 0]

    cont = 0
    for i,e1,e2 in mesh.boundary_nodes:
        if i == iant:
            cont += 1
            bound_middle[i] = cont
        else:
            cont = 1
        iant = i

    edge_labels = {}
    cont = 0
    for i,e1,e2 in mesh.boundary_nodes:
        cont += 1
        if cont == int(bound_middle[i]/2.0):
            edge_labels[e1, e2] = str(i)
        if cont == bound_middle[i]:
            cont = 0

    positions = nx.get_node_attributes(G, 'posxy')

    nx.draw_networkx(G, positions, node_size=1.5, node_color='k', font_size=0)
    nx.draw_networkx_edge_labels(G, positions, edge_labels, label_pos=0.5,
                                 font_size=10)

    plt.axes().set_aspect('equal')


    #limits=plt.axis('off')


def boundary_condition_dirichlet(displacement, mesh, dpi):

    plt.figure('Elements', dpi=dpi)

    h = mesh.AvgLength


    for line in displacement(1,1).keys():
        d= displacement(1,1)[line]
        if d[0] == 0.0 and d[1]== 0.0:
            for l, n1, n2 in mesh.boundary_nodes:
                if line == l:
                    x1 = mesh.nodes_coord[n1, 0]
                    y1 = mesh.nodes_coord[n1, 1]
                    plt.annotate('', xy=(x1, y1), xycoords='data',
                                 xytext=(x1-h,y1-h), textcoords='data',
                                 arrowprops=dict(facecolor='black', width=0.2,
                                                 headwidth=0.2))

                    x2 = mesh.nodes_coord[n2, 0]
                    y2 = mesh.nodes_coord[n2, 1]
                    plt.annotate('', xy=(x2, y2), xycoords='data',
                                 xytext=(x2-h,y2-h), textcoords='data',
                                 arrowprops=dict(facecolor='black', width=0.2,
                                                 headwidth=0.2))

    plt.draw()


def boundary_condition_neumann_value(traction, mesh, dpi):

    plt.figure('Elements', dpi=dpi)

    h = mesh.AvgLength


    for line in traction(1,1).keys():
        t = traction(1,1)[line]
        if t[0] != 0.0 or t[1] != 0.0:
            for l, n1, n2 in mesh.boundary_nodes:
                if line == l:
                    x1 = mesh.nodes_coord[n1, 0]
                    y1 = mesh.nodes_coord[n1, 1]
                    t1 = traction(x1, y1)[line]
                    t_r1 = np.sqrt(t1[0]**2.+t1[1]**2.)
                    plt.annotate(str(t_r1), xy=(x1, y1), xycoords='data',
                                 xytext=(x1 - t1[0], y1 - t1[1]),
                                 textcoords='data', size=8,
                                 verticalalignment='top',
                                 arrowprops=dict(facecolor='black', width=0,
                                                 headwidth=5, shrink=0.1))

                    t_r2 = np.sqrt(t1[0]**2.+t1[1]**2.)
                    x2 = mesh.nodes_coord[n2, 0]
                    y2 = mesh.nodes_coord[n2, 1]
                    t2 = traction(x2, y2)[line]
                    plt.annotate(str(t_r2), xy=(x2, y2), xycoords='data',
                                 xytext=(x2 - t2[0], y2 - t2[1]),
                                 textcoords='data', size=8,
                                 verticalalignment='top',
                                 arrowprops=dict(facecolor='black', width=0,
                                                 headwidth=5, shrink=0.1))

    plt.draw()


def boundary_condition_neumann(traction, mesh, dpi):

    plt.figure('Elements', dpi=dpi)

    h = mesh.AvgLength


    for line in traction(1,1).keys():
        t = traction(1,1)[line]
        if t[0] != 0.0 or t[1] != 0.0:
            for l, n1, n2 in mesh.boundary_nodes:
                if line == l:
                    x1 = mesh.nodes_coord[n1, 0]
                    y1 = mesh.nodes_coord[n1, 1]
                    t1 = traction(x1, y1)[line]
                    t_r1 = np.sqrt(t1[0]**2.+t1[1]**2.)
                    plt.annotate('', xy=(x1, y1), xycoords='data',
                                 xytext=(x1 - t1[0], y1 - t1[1]),
                                 textcoords='data', size=8,
                                 verticalalignment='top',
                                 arrowprops=dict(facecolor='black', width=0,
                                                 headwidth=5, shrink=0.1))

                    t_r2 = np.sqrt(t1[0]**2.+t1[1]**2.)
                    x2 = mesh.nodes_coord[n2, 0]
                    y2 = mesh.nodes_coord[n2, 1]
                    t2 = traction(x2, y2)[line]
                    plt.annotate('', xy=(x2, y2), xycoords='data',
                                 xytext=(x2 - t2[0], y2 - t2[1]),
                                 textcoords='data', size=8,
                                 verticalalignment='top',
                                 arrowprops=dict(facecolor='black', width=0,
                                                 headwidth=5, shrink=0.1))

    plt.draw()



def nodes_network_deformedshape_contour(mesh, a, dpi):
    c = mesh.nodes_coord

    plt.figure('Deformation Contour', dpi=dpi)
    bn = mesh.boundary_nodes

    adX = a[::2]
    adY = a[1::2]

    X, Y = c[bn[:, 1], 0], c[bn[:, 1], 1]
    dX, dY = c[bn[:, 1], 0] + adX[bn[:, 1]], c[bn[:, 1], 1] + adY[bn[:, 1]]

    G = nx.Graph()

    label = []
    for i in range(len(X)):
        label.append(i)
        G.add_node(i, posxy=(X[i], Y[i]))

    if mesh.gmsh == 1.0:
        temp = np.copy(mesh.ele_conn[:, 2])
        mesh.ele_conn[:, 2] = mesh.ele_conn[:, 3]
        mesh.ele_conn[:, 3] = temp
        mesh.gmsh += 1.0

    for i in range(len(bn[:, 0]) - 1):
       G.add_edge(i, i+1)

    G.add_edge(len(bn[:, 0]) - 1, 0)

    positions = nx.get_node_attributes(G, 'posxy')

    nx.draw_networkx(G, positions, node_size=0, node_color='k', font_size=0,
                     style = 'dashed', width=0.5)
    G2 = nx.Graph()

    label2 = []
    for i in range(len(dX)):
        label2.append(i)
        G2.add_node(i, posxy2=(dX[i], dY[i]))

    for i in range(len(bn[:, 0]) - 1):
       G2.add_edge(i, i+1)

    G2.add_edge(len(bn[:, 0]) - 1, 0)

    positions2 = nx.get_node_attributes(G2, 'posxy2')

    nx.draw_networkx(G2, positions2, node_size=0, node_color='k',
                        font_size=0)

    plt.axes().set_aspect('equal')
    #limits=plt.axis('off')


def nodes_network_edges_element_label(mesh, dpi):
    c = mesh.nodes_coord

    X, Y = c[:, 0], c[:, 1]
    plt.figure('Elements', dpi=dpi)
    G = nx.Graph()

    label = []
    for i in range(len(X)):
        label.append(i)
        G.add_node(i, posxy=(X[i], Y[i]))

    if mesh.gmsh == 1.0:
        temp = np.copy(mesh.ele_conn[:, 2])
        mesh.ele_conn[:, 2] = mesh.ele_conn[:, 3]
        mesh.ele_conn[:, 3] = temp
        mesh.gmsh += 1

    for i in range(len(mesh.ele_conn)):
        G.add_cycle([mesh.ele_conn[i, 0],
                     mesh.ele_conn[i, 1],
                     mesh.ele_conn[i, 3],
                     mesh.ele_conn[i, 2]], )

    bound_middle = {}
    iant = mesh.boundary_nodes[0, 0]

    cont = 0
    for i,e1,e2 in mesh.boundary_nodes:
        if i == iant:
            cont += 1
            bound_middle[i] = cont
        else:
            cont = 1
        iant = i

    edge_labels = {}
    cont = 0
    for i,e1,e2 in mesh.boundary_nodes:
        cont += 1
        if cont == int(bound_middle[i]/2.0):
            edge_labels[e1, e2] = str(i)
        if cont == bound_middle[i]:
            cont = 0

    positions = nx.get_node_attributes(G, 'posxy')

    nx.draw_networkx(G, positions, node_size=1.5, node_color='blue',
                     font_size=0)
    nx.draw_networkx_edge_labels(G, positions, edge_labels, label_pos=0.5,
                                 font_size=10)

    for e in range(len(mesh.ele_conn)):
        x_element = (mesh.nodes_coord[mesh.ele_conn[e, 0], 0] +
                  mesh.nodes_coord[mesh.ele_conn[e, 1], 0] +
                          mesh.nodes_coord[mesh.ele_conn[e, 2], 0] +
                          mesh.nodes_coord[mesh.ele_conn[e, 3], 0])/4.

        y_element = (mesh.nodes_coord[mesh.ele_conn[e, 0], 1] +
                          mesh.nodes_coord[mesh.ele_conn[e, 1], 1] +
                          mesh.nodes_coord[mesh.ele_conn[e, 2], 1] +
                          mesh.nodes_coord[mesh.ele_conn[e, 3], 1])/4.


        plt.annotate(str(e), (x_element, y_element), size=10, color='r')


    plt.axes().set_aspect('equal')


    #limits=plt.axis('off')


def draw_elements_label(mesh, dpi):

    for e in range(len(mesh.ele_conn)):
        x_element = (mesh.nodes_coord[mesh.ele_conn[e, 0], 0] +
                  mesh.nodes_coord[mesh.ele_conn[e, 1], 0] +
                          mesh.nodes_coord[mesh.ele_conn[e, 2], 0] +
                          mesh.nodes_coord[mesh.ele_conn[e, 3], 0])/4.

        y_element = (mesh.nodes_coord[mesh.ele_conn[e, 0], 1] +
                          mesh.nodes_coord[mesh.ele_conn[e, 1], 1] +
                          mesh.nodes_coord[mesh.ele_conn[e, 2], 1] +
                          mesh.nodes_coord[mesh.ele_conn[e, 3], 1])/4.


        plt.annotate(str(e), (x_element, y_element), size=9, color='r')


def draw_edges_label(mesh, dpi):
    c = mesh.nodes_coord

    X, Y = c[:, 0], c[:, 1]
    plt.figure('Elements', dpi=dpi)
    G = nx.Graph()


    label = []
    for i in range(len(X)):
        label.append(i)
        G.add_node(i, posxy=(X[i], Y[i]))

    bound_middle = {}
    iant = mesh.boundary_nodes[0, 0]

    cont = 0
    for i,e1,e2 in mesh.boundary_nodes:
        if i == iant:
            cont += 1
            bound_middle[i] = cont
        else:
            cont = 1
        iant = i

    edge_labels = {}
    cont = 0
    for i,e1,e2 in mesh.boundary_nodes:
        cont += 1
        if cont == int(bound_middle[i]/2.0):
            edge_labels[e1, e2] = str(i)
        if cont == bound_middle[i]:
            cont = 0

    positions = nx.get_node_attributes(G, 'posxy')

    nx.draw_networkx_edge_labels(G, positions, edge_labels, label_pos=0.5,
                                 font_size=9, font_color='b')




def draw_nodes_label(mesh, dpi):
    c = mesh.nodes_coord

    X, Y = c[:, 0], c[:, 1]
    plt.figure('Elements', dpi=dpi)
    G = nx.Graph()

    label = {}
    for i in range(len(X)):
        label[i] = i
        G.add_node(i, posxy=(X[i], Y[i]))


    positions = nx.get_node_attributes(G, 'posxy')

    nx.draw_networkx_nodes(G, positions, node_color='w', node_size=200)
    nx.draw_networkx_labels(G,positions,label,font_size=9)

