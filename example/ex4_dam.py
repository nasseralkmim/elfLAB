__author__ = 'Nasser'

import numpy as np
import elasticity2d

meshName = 'barragem1'

material = {'E':34473.786*1e6, 'nu':0.1}

def body_forces(x1, x2):
    return np.array([
        0.0,
        -24357.0,
    ])

gma = 9820.0*10
def traction_imposed(x1, x2):
    return {
        ('line', 5):[(x2 <= 116.13)*(-gma*(x2-116.13)), 0.0],
        ('line', 6):[(17.07*gma - 116.13*gma)*x2/99.06 + 116.13*gma,
                     -((17.07*gma - 116.13*gma)*x2/99.06 + 116.13*gma)]
    }

def displacement_imposed(x1, x2):
    return {
    ('line', 0):[0.0, 0.0]

    }

elasticity2d.solver(meshName,
                              material,
                              body_forces,
                              traction_imposed,
                              displacement_imposed,
                              plotUndeformed={'Domain':True,
                                              'Elements':True,
                                              'NodeLabel':False,
                                              'EdgesLabel':False,
                                              'ElementLabel':False},
                              plotStress={'s11':False,
                                          's22':False,
                                          's12':False,
                                          'sPmax':True,
                                          'sPmin':True},
                              plotDeformed={'DomainUndeformed':True,
                                            'ElementsUndeformed':False,
                                            'DomainDeformed':True,
                                            'ElementsDeformed':False,
                                            'DeformationMagf': 1000}
                              )


