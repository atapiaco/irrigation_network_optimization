# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 10:23:02 2023

@author: atapi
"""

import INO_WaterNetworkFunctions as WNF
import INO_WaterNetworkConstants as WNC
import INO_WaterNetworkPlots     as WNP

import matplotlib.pyplot         as plt
import pandas                    as pd
import numpy                     as np
import INO_GeneticOperators      as GenOps

import csv

def plot_evolucion(c,m,it):
    
    fileName = 'Resultados\RES_SO\LOG_' + str(c) + '_' + str(m) + '_IT' + str(it) + '.csv'
    file     = open(fileName, 'r')
    
    gen      = []
    fit_ave  = []
    fit_mins = []
    fit_maxs = []
    
    data = list(csv.reader(file, delimiter=","))
    file.close()
    
    for i,j in enumerate(data):
        if i==0:
            continue
        gen.append(float(j[0]))
        fit_ave.append(float(j[2]))
        fit_mins.append(float(j[4]))
        fit_maxs.append(float(j[5]))            
    
    
    fig, ax1 = plt.subplots()
    ax1.plot(gen, fit_mins, "b")
    # ax1.plot(gen, fit_maxs, "r")
    ax1.plot(gen, fit_ave, "--k")
    # ax1.fill_between(gen, fit_mins, fit_maxs, facecolor='g', alpha = 0.2)
    # ax1.fill_between(gen, fit_mins, fit_maxs, where=fit_maxs >= fit_mins, facecolor='g', alpha = 0.2)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.set_ylim([min(fit_mins)*0.98, max(fit_mins)*1.05])
    ax1.legend(["Min", "Max", "Avg"], loc="Upper center")
    plt.grid(True)
    # plt.savefig("Convergencia.eps", dpi = 300)


def extractInd(c,m,k):    
    fileName = 'Resultados\RES_SO\IND_' + str(c) + '_' + str(m) + '.txt'
    file     = open(fileName, 'r')
    data     = list(csv.reader(file, delimiter=","))
    file.close()
    ind0 = [ int(j.replace(']','').replace('[','')) for i,j in enumerate(data[k])][1:]
    ind  = [ [ind0[i+2],ind0[i+3]] for i in range(len(ind0)-2) if not i%2 ]
    ind.insert(0,0)
    return ind
        
def findBestFit(c,m):
    bestFit   = [100,0]
    fileName = 'Resultados\RES_SO\FIT_' + str(c) + '_' + str(m) + '.txt'
    file     = open(fileName, 'r')
    data     = list(csv.reader(file, delimiter=","))
    file.close()
    fitValues = [ float(j[4]) for i,j in enumerate(data)]
    bestFit    = min(fitValues)
    pos       = fitValues.index(bestFit)
    return bestFit, pos


from statistics import stdev, mean

def optPopStats():    
    print('-'*90)
    print(' c ','\t  m ','\t   Min','\t\t   Max','\t\t   Avg','\t\t   Std')
    print('-'*90)
    valores_c = [ 0.7, 0.6, 0.5, 0.4, 0.3 ]
    valores_m = [ 0.3, 0.4, 0.5, 0.6, 0.7 ]
    for c,m in zip(valores_c, valores_m):         
        fileName = 'Resultados\RES_SO\FIT_' + str(c) + '_' + str(m) + '.txt'
        file     = open(fileName, 'r')
        data     = list(csv.reader(file, delimiter=","))
        file.close()
        
        bestFit, pos = findBestFit(c,m)
        ind = extractInd(c,m,pos)
        Q = WNF.ObtainIrrigationFlow(WNF.SolveNetwork(ind[1:],0))*3.6e6  
        C = WNF.PipeCost(ind[1:],0)
        
        fitnessValues = [float(j[4]) for i,j in enumerate(data)]
        fit_min = min(fitnessValues)
        fit_max = max(fitnessValues)
        fit_std = stdev(fitnessValues)
        fit_avg = mean(fitnessValues)
        
        print(c,'\t',m,'\t','{0:.6f}'.format(fit_min),
                  '\t','{0:.6f}'.format(fit_max),
                  '\t','{0:.6f}'.format(fit_avg),
                  '\t','{0:.6f}'.format(fit_std))
    print('-'*90)
    
def optIndStats():    
    print('-'*90)
    print(' c ','\t  m ','\t  L   ', '\t\t dQ', '\t\t Qav', '\t\t Qsd')
    print('-'*90)
    valores_c = [ 0.7, 0.6, 0.5, 0.4, 0.3 ]
    valores_m = [ 0.3, 0.4, 0.5, 0.6, 0.7 ]
    for c,m in zip(valores_c, valores_m):        
        bestFit, pos = findBestFit(c,m)
        ind = extractInd(c,m,pos)
        Q = WNF.ObtainIrrigationFlow(WNF.SolveNetwork(ind[1:],0))*3.6e6  
        C = WNF.PipeCost(ind[1:],0)
       
        print(c,'\t',m,'\t', '{0:.2f}'.format(C), '\t{0:.4f}'.format(max(Q)-min(Q)), '\t\t{0:.4f}'.format(mean(Q)), '\t\t{0:.4f}'.format(stdev(Q)) )
        
    print('-'*90)
        
# %%:

c = 0.5
k = 3

plot_evolucion(c, 1-c, k)

# %% PROBANDO INDIVIDUOS

print('\n'*2)
optPopStats()
print('\n'*2)
optIndStats()
print('\n'*2)

# %%

fit1, pos = findBestFit(0.7, 0.3)


ind0 = WNC.indRef[:]
ind1 = extractInd(0.7, 0.3, pos)

X0 = WNF.SolveNetwork(ind0[1:],0)
X1 = WNF.SolveNetwork(ind1[1:],0)
C0 = WNF.PipeCost(ind0[1:],0)
C1 = WNF.PipeCost(ind1[1:],0)
Q0 = WNF.ObtainIrrigationFlow(X0)*3.6e6
Q1 = WNF.ObtainIrrigationFlow(X1)*3.6e6

R0 = WNF.RelFlowDeviation(X0)
R1 = WNF.RelFlowDeviation(X1)

print('\n'*5)
print('-'*50)
print(' '*5,' REFERENCE SOLUTION ')
print('-'*50)

print('Fitness ', R0)
print('dQmax:  ', max(Q0)-min(Q0))
print('Qmean:  ', mean(Q0))
print('Qstdev: ', stdev(Q0))
print('Length: ', C0)

print('-'*50)
print(' '*5,' OPTIMAL SOLUTION ')
print('-'*50)

print('Fitness ', R1)
print('dQmax:  ', max(Q1)-min(Q1))
print('Qmean:  ', mean(Q1))
print('Qstdev: ', stdev(Q1))
print('Length: ', C1)

print('-'*50)
print(' '*5,' COMPARISON ')
print('-'*50)

print('Qmin ref: ',min(Q0),' L/h')
print('Qmin opt: ',min(Q1),' L/h')
t0 = 5/min(Q0)
t1 = 5/min(Q1)
print('Time ref: ',t0,' h')
print('Time opt: ',t1,' h')
V0 = sum([Q0[i]*t0-5 for i in range(len(Q0))])
V1 = sum([Q1[i]*t1-5 for i in range(len(Q1))])
print('Excess ref: ',V0*365,' L/year')
print('Excess ref: ',V1*365,' L/year')
print('Excess reduction: ',(V0-V1)/V0*100,' %')



print('-'*50)

print('\n'*10)


plt.close('all')





fig, ax = plt.subplots()
WNP.PlotWaterNetwork(ind0[1:])
ax.set_aspect(.8)

fig, ax = plt.subplots()
WNP.PlotWaterNetwork(ind1[1:])
ax.set_aspect(.8)

fig, ax = plt.subplots()
plt.plot(sorted(Q0),'o')
plt.plot(sorted(Q1),'o')
plt.legend(['Reference','Generated'])
plt.xlabel("Dripper")
plt.ylabel("Irrigation flow rate, Q (L/s)")

fig, ax = plt.subplots()
plt.boxplot([Q0,Q1], notch=None, vert=None, patch_artist=None, widths=None)
plt.ylabel("Irrigation flow rate, Q (L/s)")
plt.xticks([1, 2], ['REFERENCE', 'OPTIMAL'])
