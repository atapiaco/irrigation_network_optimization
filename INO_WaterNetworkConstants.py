# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:16:03 2023

@author: atapi
"""
import networkx as nx
import numpy    as np

# %% TERRAIN DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Terrain height function
# Domain is x,y in [-50, 50] meters
# def Fheight(x, y):
#     x = x/30
#     # y = y/20
#     return ( (x**2 + y - 11)**2 + (x + y**2 -7)**2 ) * 0.02

# def Fheight(x, y):
#     # x = x/40
#     # y = -(y+20)/40
#     x = -x/50
#     y = (y)/50
#     return ( (x**2 + y - 11)**2 + (x + y**2 -7)**2 ) * 0.04


def Fheight(x,y):
    x = x/50
    y = y/50
    return 3*(1-x)**2*2.718**(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*2.718**(-x**2-y**2) - 1/3*2.718**(-(x+1)**2 - y**2)



# Domain
terrain_limits = [-50, 50, -50, 50]

# Nodes in the terrain
# Nnod = 20
# nod  = [ [(terrain_limits[1]-terrain_limits[0])*random.random()+terrain_limits[0],
#           (terrain_limits[3]-terrain_limits[2])*random.random()+terrain_limits[2] ] for i in range(Nnod)]
# hgt  = [Fheight(j[0],j[1]) for j in nod]

N = 7
M = 7

nod = []
xx = np.linspace(terrain_limits[0], terrain_limits[1], N)
yy = np.linspace(terrain_limits[2], terrain_limits[3], M)
for i in range(N):
    for j in range(M):
        nod.append([xx[i], yy[j]])
Nnod = len(nod)
hgt  = [Fheight(j[0],j[1]) for j in nod]

indRef = [0]
for i in range(M-1):
    indRef.append([i,i+1])
    
for i in range(M):
    for j in range(N-1):
        indRef.append([i+j*M, i+(j+1)*M])
        
# print(indRef)

# plt.style.use('seaborn-white')
# x    = np.linspace(terrain_limits[0], terrain_limits[1], 100)
# y    = np.linspace(terrain_limits[2], terrain_limits[3], 100)
# X, Y = np.meshgrid(x, y)
# Z    = Fheight(X, Y)

# plt.imshow(Z, extent=terrain_limits, origin='lower',
#            cmap='rainbow', alpha=1)
# plt.colorbar()
# contours = plt.contour(X, Y, Z, 10, colors='black')
# plt.clabel(contours, inline=False, fontsize=12)


# %% ENGINEERING CONSTANTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Friction factor of emitters
Kd = .398/3.6e6
# Pipe diameter and friction coefficient
Dp = 8e-3
# Constant for Hazen-Williams
Cp = 100
# Constant for Darcy-Weishbach
fp = 0.035

# Pipe cost (per meter)
PipeMeterCost = 1

# Water pump constants
# Function in the form  H = Hb - Kb Q^2
# Hb =  20.4
# Kb = -2.58e3
Hb =  120
Kb = -.1e3

# Nominal flow required
ReqFlow = [2e-7 for i in range(Nnod)]


# %% ALGORITHM DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Number of iterations
Nit = 6

# Tolerance for relative variance of irrigated flow
RelFlowDiff_tol = 0.01

# Maximum cost allowed for cost constrained problem
MaxCostAllowed =  8216.38



# %% EDGES LIST AND GRAPHS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# All possible edges
AllEdges = [[i, j] for i in range(Nnod) for j in range(i+1,Nnod) ]

# All possible edges belonging to Delaunay triangulation
from scipy.spatial import Delaunay
tri = Delaunay(nod)
AllEdgesDelaunay = []
delaunay = []
for i in tri.simplices:
    delaunay.append(list([i[0],i[1]]))
    delaunay.append(list([i[1],i[2]]))
for i,j in enumerate(delaunay):
    if j[0]>j[1]:
        delaunay[i][0], delaunay[i][1] = j[1], j[0]
for edge in delaunay:
    if edge not in AllEdgesDelaunay:
        AllEdgesDelaunay.append(edge)
        
# Graph with Delaunay edges
GD = nx.Graph()
AllEdgesDelaunay_Weighted = [ [min(j), max(j), 1] for i,j in enumerate(AllEdgesDelaunay) ]
GD.add_weighted_edges_from(AllEdgesDelaunay_Weighted)

# %% 