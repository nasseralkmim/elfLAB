__author__ = 'Nasser'

import heat2d

meshName = 'patch'

material= {'k-s1':1.0}

def internal_heat(x1, x2):
    return 0.0

def flux_imposed(x1, x2):
    return {
        1:10.0,
        2:0.0,
        0:0.0
    }

def temperature_imposed(x1, x2):
    return {
            3:32.0,

            }

heat2d.solver(meshName,
              material,
              internal_heat,
              flux_imposed,
              temperature_imposed,
              plotUndeformed={'Domain':True,
                              'Elements':False,
                              'NodeLabel':False,
                              'EdgesLabel':True,
                              'ElementLabel':False},
              plotTemperature={'Contour':True}
              )


