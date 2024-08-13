# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:23:19 2023

@author: atapi
"""

import INO_WaterNetworkConstants as WNC

import math
import numpy as np

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
