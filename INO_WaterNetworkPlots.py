# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 12:51:15 2023

@author: atapi
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import INO_WaterNetworkConstants as WNC


def PlotTerrain():    
    plt.style.use('seaborn-white')
    x    = np.linspace(WNC.terrain_limits[0], WNC.terrain_limits[1], 100)
    y    = np.linspace(WNC.terrain_limits[2], WNC.terrain_limits[3], 100)
    X, Y = np.meshgrid(x, y)
    Z    = WNC.Fheight(X, Y)
    
    plt.imshow(Z, extent=WNC.terrain_limits, origin='lower',
               cmap='rainbow', alpha=1)
    plt.colorbar()
    contours = plt.contour(X, Y, Z, 10, colors='grey')
    plt.clabel(contours, inline=False, fontsize=10)
    
    plt.show()    

def PlotWaterNetwork(CONNECT):    
    PlotTerrain()
    
    G = nx.DiGraph()
    G.add_nodes_from(np.arange(1,WNC.Nnod))
    G.add_edges_from(CONNECT)
    
    pos = {i:(j[0], j[1]) for i, j in enumerate(WNC.nod)}
    nx.draw_networkx(G, pos=pos,
                     node_shape='o',
                     arrows=False,
                     node_color='black',
                     node_size = 100,
                     font_color = 'white',
                     linewidths=10,
                     width=2)
    plt.show()
    
def PlotPumpCurve():
    qmax = (-WNC.Hb/WNC.Kb)**.5
    qq   = np.linspace(0,qmax,1000)
    hh   = WNC.Hb + WNC.Kb*qq**2
    plt.plot(qq*1000,hh)
    
    plt.xlabel('Caudal, Q (L/h)')
    plt.ylabel('Altura, H (m)')
    
    plt.grid(True)

# %% EJEMPLOS

# IND = [0, [0, 1], [1, 2], [2, 3], [0, 3]]

# PlotWaterNetwork(IND[1:])
