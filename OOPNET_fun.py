# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:44:34 2024

@author: atapi
"""

coord = [[-50.0, -50.0],
         [-50.0, -33.33],
         [-50.0, -16.66],
         [-50.0,  0.0],
         [-50.0,  16.66],
         [-50.0,  33.33],
         [-50.0,  50.0],
         [-33.33, -50.0],
         [-33.33, -33.33],
         [-33.33, -16.67],
         [-33.33, 0.0],
         [-33.33, 16.67],
         [-33.33, 33.33],
         [-33.33, 50.0],
         [-16.67, -50.0],
         [-16.67, -33.33],
         [-16.67, -16.67],
         [-16.67, 0.0],
         [-16.67, 16.67],
         [-16.67, 33.33],
         [-16.67, 50.0],
         [0.0, -50.0],
         [0.0, -33.33],
         [0.0, -16.67],
         [0.0, 0.0],
         [0.0, 16.67],
         [0.0, 33.33],
         [0.0, 50.0],
         [16.67, -50.0],
         [16.67, -33.33],
         [16.67, -16.67],
         [16.67, 0.0],
         [16.67, 16.67],
         [16.67, 33.33],
         [16.67, 50.0],
         [33.33, -50.0],
         [33.33, -33.33],
         [33.33, -16.67],
         [33.33, 0.0],
         [33.33, 16.67],
         [33.33, 33.33],
         [33.33, 50.0],
         [50.0, -50.0],
         [50.0, -33.33],
         [50.0, -16.67],
         [50.0, 0.0],
         [50.0, 16.67],
         [50.0, 33.33],
         [50.0, 50.0]]

hgt =  [ 1.8558,    2.4901,    1.4991,   -0.7243,   -2.2175,   -1.7686,   -0.2730,
         1.5391,    3.3788,    3.5481,    1.7615,    0.1059,    0.3066,    1.5727,
        -0.1147,    2.2007,    3.0732,    1.7771,    0.4444,    1.0299,    2.6148,
        -1.6523,    0.6227,    1.8073,    0.9811,    0.1175,    1.1028,    2.9372,
        -2.1722,   -0.2751,    0.8857,    0.4342,   -0.0530,    0.9763,    2.6419,
        -1.5158,    0.0125,    1.0854,    0.9523,    0.6237,    1.2115,    2.1940,
         0.2290,    1.9628,    3.3987,    3.6890,    3.1959,    2.7399,    2.4343]

# .15 .004 9 = 1.68
RO  = 0.15
KE  = 0.004
DIA = 9
HB  = 60
KB  = 1

qq = [i*0.01 for i in range(600)]
hh = [HB-KB*j**2 for j in qq]



import matplotlib.pyplot as plt
import oopnet as on
from statistics import stdev, mean

plt.plot(qq,hh)

def FitnessOOPNET(ind, depurar=False):

    network = on.Network()
    
    # NODOS
    for i in range(len(coord)):
        on.add_junction(network=network,
                        junction=on.Junction(id='J'+str(i+1),
                                             elevation=hgt[i],
                                             xcoordinate=coord[i][0],
                                             ycoordinate=coord[i][1],
                                             emittercoefficient=KE,
                                             demand=0))
    # TUBERÍAS
    for i,j in enumerate(ind[1:]):
        n1, n2 = j
        x1,y1  = coord[n1]
        x2,y2  = coord[n2]
        length = ((x1-x2)**2 + (y1-y2)**2)**0.5
        on.add_pipe(network=network,
                    pipe=on.Pipe(id='P'+str(i+1),
                                 length=length,
                                 diameter=DIA,
                                 roughness=RO,
                                 startnode=on.get_node(network, 'J'+str(n1+1)),
                                 endnode=on.get_node(network, 'J'+str(n2+1))))
    
    # RESERVOIR
    on.add_element.add_reservoir(network=network,
                                 reservoir=on.Reservoir(id='R',
                                                        xcoordinate=-70,
                                                        ycoordinate=-50,
                                                        head=0))
    
    # CURVA
    on.add_curve(network=network,
                 curve=on.Curve(id='C1',
                                xvalues=qq,
                                yvalues=hh))
                 
    
    
    # BOMBA
    on.add_element.add_pump(network=network, 
                            pump=on.Pump(id='PUMP',
                                         startnode=on.get_node(network, 'R'),
                                         endnode=on.get_node(network, 'J'+str(ind[0]+1)),
                                         status='open',
                                         head=on.get_curve(network=network,
                                                           id='C1')))
    
    network.options.units       = 'CMH'
    network.options.headloss    = 'D-W'
    network.options.demandmodel = 'PDA'
    
    report = network.run()
    
    network.write('PROBANDO_ESTO_1')
    
    if all(report.pressure>=0):
        caudal = [KE*j**0.5 if j >0 else 0 for j in report.pressure]
        fit = stdev(caudal)/mean(caudal)
    else:
        fit = 1e3
        
    if depurar:
        return report
    else:
        return fit,






