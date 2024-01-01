# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:23:19 2023

@author: atapi
"""

import INO_WaterNetworkFunctions as WNF
import INO_WaterNetworkConstants as WNC

import random
import networkx as nx
import matplotlib.pyplot as plt

PENALIZA = 1E5

# %% ORIGINAL OPERATORS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Generation of feasible individuals
def IndividualGeneration():
    CONNECT = []
    AvailableEdges = WNC.AllEdges[:]
    while not WNF.IsValid(CONNECT) or not WNF.IsConnected(CONNECT):
        NewEdge = random.choice(AvailableEdges)
        CONNECT.append(NewEdge)
        AvailableEdges.remove(NewEdge)
    # while random.random() < 0.5 and AvailableEdges != []:
    #     NewEdge = random.choice(AvailableEdges)
    #     CONNECT.append(NewEdge)
    #     AvailableEdges.remove(NewEdge)
    PUMPNODE = [0]
    return PUMPNODE + CONNECT

# Generation of feasible individuals
def IndividualGenerationDelaunay():
    CONNECT = []
    AvailableEdges = WNC.AllEdgesDelaunay[:]
    while not WNF.IsValid(CONNECT) or not WNF.IsConnected(CONNECT):
        NewEdge = random.choice(AvailableEdges)
        CONNECT.append(NewEdge)
        AvailableEdges.remove(NewEdge)
    # while random.random() < 0.5 and AvailableEdges != []:
    #     NewEdge = random.choice(AvailableEdges)
    #     CONNECT.append(NewEdge)
    #     AvailableEdges.remove(NewEdge)
    PUMPNODE = [0]
    return PUMPNODE + CONNECT

# Fitness Function based on MaxFlowDiff
def Fitness_MaxFlowDiff(IND):
    PUMPNODE = IND[0 ]
    CONNECT  = IND[1:]
    if not (WNF.IsValid(CONNECT) and WNF.IsConnected(CONNECT)):
        return PENALIZA,
    X = WNF.SolveNetwork(CONNECT, PUMPNODE)
    WNF.RelFlowDeviation(X)
    return WNF.MaximumFlowDifference(CONNECT, PUMPNODE)
    
def Fitness_RelFlowDev_old(IND):
    PUMPNODE = IND[0 ]
    CONNECT  = IND[1:]    
    if WNF.IsValid(CONNECT) and WNF.IsConnected(CONNECT):
        X = WNF.SolveNetwork(CONNECT, PUMPNODE)
        if WNF.IsFlowPositive(X):
            return WNF.RelFlowDeviation(X),
        else:
            errorCode = 2
    else:
        errorCode = 1
    return PENALIZA + errorCode,

def Fitness_RelFlowDev(IND):
    PUMPNODE = IND[0 ]
    CONNECT  = IND[1:]    
    X = WNF.SolveNetwork(CONNECT, PUMPNODE)
    if WNF.IsFlowPositive(X):
        return WNF.RelFlowDeviation(X),
    else:
        return PENALIZA,
    
# Fitness Function based on Cost
def Fitness_Cost(IND):
    PUMPNODE = IND[0 ]
    CONNECT  = IND[1:]
    if WNF.IsValid(CONNECT) and WNF.IsConnected(CONNECT):
        X = WNF.SolveNetwork(CONNECT, PUMPNODE)
        if WNF.IsFlowPositive(X) and WNF.RelFlowDeviation(X) < 0.5 * 0.069:
            return WNF.PipeCost(CONNECT, PUMPNODE),
    return PENALIZA,

def Fitness_FlowDev_and_Cost(IND):
    PUMPNODE = IND[0 ]
    CONNECT  = IND[1:]
    if WNF.IsValid(CONNECT) and WNF.IsConnected(CONNECT):
        X = WNF.SolveNetwork(CONNECT, PUMPNODE)
        if WNF.IsFlowPositive(X):
            return WNF.RelFlowDeviation(X), WNF.PipeCost(CONNECT, PUMPNODE),
    return PENALIZA, PENALIZA

# Mutation of an individual
# It can lose an edge or receive a new one
def Mutation(IND, P01=0.5, P10=0.5):
    AvailableEdges = WNC.AllEdges[:]
    for edge in IND[1:]:
        AvailableEdges.remove(edge)
    if random.random() < P01 and AvailableEdges != []:        
        NewEdge = random.choice(AvailableEdges)
        IND.append(NewEdge)
        AvailableEdges.remove(NewEdge)
    if random.random() < P10 and len(IND)>1:
        RemovedEdge = random.choice(IND[1:])
        IND.remove(RemovedEdge)
        AvailableEdges.append(RemovedEdge)
    while not WNF.IsValid(IND[1:]) or not WNF.IsConnected(IND[1:]):
        NewEdge = random.choice(AvailableEdges)
        IND.append(NewEdge)
        AvailableEdges.remove(NewEdge)
    return IND,

# Mutation of an individual using DELAUNAY edges
# It can lose an edge or receive a new one
def MutationDelaunay(IND, P01=0.1, P10=0.1):
    AvailableEdges = WNC.AllEdgesDelaunay[:]
    for edge in IND[1:]:
        AvailableEdges.remove(edge)
    if random.random() < P01 and AvailableEdges != []:        
        NewEdge = random.choice(AvailableEdges)
        IND.append(NewEdge)
        AvailableEdges.remove(NewEdge)
    if random.random() < P10 and len(IND)>1:
        RemovedEdge = random.choice(IND[1:])
        IND.remove(RemovedEdge)
        AvailableEdges.append(RemovedEdge)
    while not WNF.IsValid(IND[1:]) or not WNF.IsConnected(IND[1:]):
        NewEdge = random.choice(AvailableEdges)
        IND.append(NewEdge)
        AvailableEdges.remove(NewEdge)
    return IND,

# Crossover of two individuals
# Each offspring is created as a copy of one of the parents extended with 
# a copy of a subset of the other parent's edges
def Crossover(IND1, IND2, P1=0.1):
    IND10 = IND1[:]
    IND20 = IND2[:]
    for edge in IND10:
        if random.random() < P1 and edge not in IND2:
            IND2.append(edge)
    for edge in IND20:
        if random.random() < P1 and edge not in IND1:
            IND1.append(edge)
    return IND1, IND2

# %% GRAPH-BASED MUTATION AND CROSSOVER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def G_IndividualGeneration():
    G = nx.Graph()
    for edge in WNC.AllEdgesDelaunay:
        G.add_edge(edge[0], edge[1], weight=random.random())
    MinSpanTree = nx.minimum_spanning_edges(G, algorithm='kruskal', data=False)    
    PUMPNODE = [0]
    CONNECT = []
    for edge in MinSpanTree:
        CONNECT.append([min(edge), max(edge)])
    return PUMPNODE + CONNECT


def G_Mutation(IND, Nmut=1):
    AvailableEdges = WNC.AllEdgesDelaunay[:]
    for edge in IND[1:]:
        if edge in AvailableEdges:
            AvailableEdges.remove(edge)
    NewEdges = random.sample(AvailableEdges, k=Nmut)
    G = nx.Graph()
    for edge in IND[1:]:
        G.add_edge(edge[0], edge[1], weight=random.random())
    for edge in NewEdges:
        G.add_edge(edge[0], edge[1], weight=0)
    MinSpanTree = nx.minimum_spanning_edges(G, algorithm='kruskal', data=False)
    del IND[1:]
    for edge in MinSpanTree:
        IND.append([edge[0], edge[1]])
    return IND,
    
def G_Crossover(IND1, IND2):
    IND10 = IND1[:]
    IND20 = IND2[:]
    del IND1[1:]
    del IND2[1:]
    
    # First offspring
    G = WNC.GD.copy()
    nx.set_edge_attributes(G, {(u, v): random.random() for u, v in G.edges()}, 'weight')
    for edge in IND10[1:]:
        G[edge[0]][edge[1]]['weight'] *= 0.5
    for edge in IND20[1:]:
        G[edge[0]][edge[1]]['weight'] *= 0.5
    MinSpanTree = nx.minimum_spanning_edges(G, algorithm='kruskal', data=False)
    for edge in MinSpanTree:
        IND1.append([min(edge), max(edge)])
        
    # Second offspring
    G = WNC.GD.copy()
    nx.set_edge_attributes(G, {(u, v): random.random() for u, v in G.edges()}, 'weight')
    for edge in IND10[1:]:
        G[edge[0]][edge[1]]['weight'] *= 0.5
    for edge in IND20[1:]:
        G[edge[0]][edge[1]]['weight'] *= 0.5
    MinSpanTree = nx.minimum_spanning_edges(G, algorithm='kruskal', data=False)
    for edge in MinSpanTree:
        IND2.append([min(edge), max(edge)])
        
    return IND1, IND2
    
    
    
    
# %% EJEMPLOS

# IND = [0,[0,1],[0,2],[1,3],[2,3]]

# WNP.PlotWaterNetwork(IND[1:])

# IND2 = Mutation(IND, P1=0.5)

# plt.figure()
# WNP.PlotWaterNetwork(IND2[1:])
            
    
    
