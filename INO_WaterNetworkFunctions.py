# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:23:19 2023

@author: atapi
"""

import INO_WaterNetworkConstants as WNC

import math
import numpy as np

def SolveNetwork(CONNECT, PUMPNODE=0):
        
    Nseg = len(CONNECT)
    Nnod = WNC.Nnod
    
    # Xold = [ 30+10*random.random() for i in range(Nnod) ] + [ random.random() for i in range(Nnod+Nseg) ]
    Xold = [ 30 for i in range(Nnod) ] + [ 1 for i in range(Nnod+Nseg) ]
    
    # El caudal de riego de cada nodo i es el Q[N+i]
    # 
    #  [ X(0) X(1) ... X(N-1)  |  X(N) X(N+1) ... X(2N-1)  |  X(2N) ... X(2N+M-1) ] 
    #  [ P1   P2   ... PN      |  W1   W2     ... WN       |  QN+1  ... QN+M      ]
    
    for iteration in range(WNC.Nit):
        
        A = []
        b = []
        
        # Conservation equations at nodes
        
        for node in range(WNC.Nnod):
            
            newline = [0 for i in range(2*Nnod+Nseg)]
            
            # Flow rates Q (indices) that arrive at NODE
            i_Qin  = [2*Nnod+i for i,j in enumerate(CONNECT) if j[1] == node]
            
            # Flow rates Q (indices) that exit NODE
            i_Qout = [2*Nnod+i for i,j in enumerate(CONNECT) if j[0] == node]
            
            # FLow rate W (index) that is generated at NODE
            i_Qout.append(Nnod + node)
            
            newline   = [-1 if i in i_Qin  else j for i,j in enumerate(newline)]
            newline   = [+1 if i in i_Qout else j for i,j in enumerate(newline)]
            newline_b = 0
            
            A.append(newline)
            b.append(newline_b)
            
        # Bernoulli at every segment
        
        for seg in range(Nseg):
            
            newline = [0 for i in range(2*Nnod+Nseg)]
            
            nod_i = CONNECT[seg][0]
            nod_j = CONNECT[seg][1]
            
            hgt_i = WNC.hgt[nod_i]
            hgt_j = WNC.hgt[nod_j]
            
            i_Pi  = nod_i
            i_Pj  = nod_j
            
            i_Qij = 2*Nnod + seg
            
            vLij  = [WNC.nod[nod_j][k]-WNC.nod[nod_i][k] for k in range(2)]
            Lij   = math.sqrt(sum([i**2 for i in vLij]))
            
            newline[i_Pi ] =  1
            newline[i_Pj ] = -1
            
            # Hazen-Williams
            # Kf_ij = 10.583 * Lij * WNC.Cp**(-1.85) * WNC.Dp**(-4.87)
            # newline[i_Qij] = -Kf_ij * abs(Xold[i_Qij])**.85
            # newline_b      =  hgt_j - hgt_i
            
            # Darcy-Weishbach
            Kf_ij = WNC.fp*8*Lij / (3.14**2*9.8*WNC.Dp**5)
            newline[i_Qij] = -Kf_ij * abs(Xold[i_Qij])
            newline_b      =  hgt_j - hgt_i
 
            A.append(newline)
            b.append(newline_b)
            
        # Dripper and Pump equations
        
        for node in range(Nnod):
            
            newline = [0 for i in range(2*Nnod+Nseg)]
            
            i_Qi = node + Nnod
            i_Pi = node
                
            if node == PUMPNODE:
                
                #  Pump equation (pseudo-dripper at PUMPNODE)
                newline[i_Qi] = -1*WNC.Kb*Xold[i_Qi]
                newline[i_Pi] =  1
                newline_b     = WNC.Hb
                
                A.append(newline)
                b.append(newline_b)
            
            else:
                
                # Dripper
                newline[i_Qi] = -1
                newline[i_Pi] = WNC.Kd * abs(Xold[i_Pi])**(-0.5)
                newline_b     = 0
                
                A.append(newline)
                b.append(newline_b)
            
        # Solve the system Ax=b
        
        # try:
        X    = np.linalg.solve(np.array(A), np.array(b))
        # except:
        #     print('\n\nIndividuo problemático: \n')
        #     print(CONNECT)
        Xold = X
    
    return X

def ObtainIrrigationFlow(X):
    Qirr = X[WNC.Nnod+1:2*WNC.Nnod]
    return Qirr

# Maximum deviation from the nominal irrigation flow rate
#that is measured at the different drippers
def MaxFlowDeviation(X): 
    Qirr = X[WNC.Nnod+1:2*WNC.Nnod]
    return max([ abs(WNC.ReqFlow[i]-Qirr[i]) for i in range(WNC.Nnod) ])

# Maximum ABSOLUTE difference in irrigation flow rate
# that appear at the different drippers
def MaxFlowDifference(X):
    Qirr = X[WNC.Nnod+1:2*WNC.Nnod]   
    return max(Qirr)-min(Qirr)

# Maximum RELATIVE difference in irrigation flow rate
# that appear at the different drippers
def RelFlowDeviation(X):  
    Qirr      = X[WNC.Nnod+1:2*WNC.Nnod]
    Qirr_mean = sum(Qirr)/len(Qirr)
    return np.std(Qirr)/Qirr_mean

# Cost of the pipe
# Length of the pipe x Cost per meter
def PipeCost(CONNECT, PUMPNODE):
    length = 0
    for edge in CONNECT:
        node1   = WNC.nod[edge[0]]
        node2   = WNC.nod[edge[1]]
        length += (sum([(i[1]-i[0])**2 for i in zip(node1,node2)]))**.5
    return WNC.PipeMeterCost*length

# Verufy that the flow at all emitters is positive
def IsFlowPositive(X):
    Qirr = X[WNC.Nnod+1:2*WNC.Nnod]
    return all([j>0 for j in Qirr])

# Verify is the resulting graph is connected
# This means that every dripper is connected to the water network
def IsConnected(CONNECT):
    # Build adjacency list representation of graph
    adj_list = {}
    for u, v in CONNECT:
        adj_list.setdefault(u, []).append(v)
        adj_list.setdefault(v, []).append(u)
    # Perform DFS to visit all nodes in the graph
    visited = set()
    stack = [1]  # Start with node 1
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(adj_list.get(node, []))
    # Check if all nodes were visited
    return len(visited) == len(adj_list)

# Check that the node IDs exist
# The maximum Node ID is Nnod-1
def IsValid(CONNECT):
    NodesInNetwork = set([i for i in range(WNC.Nnod)])
    NodesContained = set(sum(CONNECT, []))
    if NodesInNetwork == NodesContained:
        return True
    else:
        return False

# %% EJEMPLOS

# IND      = [0, [0, 1], [1, 2], [2, 3], [0, 3]]
# # IND      = [0, [1, 3]]
# CONNECT  = IND[1:]
# PUMPNODE = IND[0]

# print('\n\n' + '*'*60 + '\n\n')

# print('El sistema es ',end='')
# print('VÁLIDO' if IsValid(CONNECT) else 'NO VÁLIDO',end='')
# print(' y CONEXO' if IsConnected(CONNECT) else ' e INCONEXO')

# if IsValid(CONNECT) and IsConnected(CONNECT):
        
#     print('\nMáxima desviación de riego requerido:')
#     print(MaxFlowDiff(CONNECT, PUMPNODE))
    
#     X    = SolveNetwork(CONNECT, PUMPNODE)   
#     Qirr = X[WNC.Nnod:2*WNC.Nnod]
    
#     print('\nPresiones en nodos:')
#     print(X[:WNC.Nnod])
    
#     print('\nCaudales en tuberías:')
#     print(X[WNC.Nnod:len(CONNECT)+WNC.Nnod:])
    
#     print('\nCaudales de riego:')
#     print(X[len(CONNECT)+WNC.Nnod:])

# print('\n\n' + '*'*60 + '\n\n')