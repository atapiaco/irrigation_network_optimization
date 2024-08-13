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
import OOPNET_fun                as OOPNET

import csv

def plot_evolucion(c,m,it):
    
    fileName = 'Resultados\RES_SO_OOPNET\OOPNET_LOG_' + str(c) + '_' + str(m) + '_IT' + str(it) + '.csv'
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
    
    print(fit_mins)
    print(fit_ave)
    fig, ax1 = plt.subplots()
    ax1.plot(gen, fit_mins, "b")
    # ax1.plot(gen, fit_maxs, "r")
    ax1.plot(gen, fit_ave, "--k")
    # ax1.fill_between(gen, fit_mins, fit_maxs, facecolor='g', alpha = 0.2)
    # ax1.fill_between(gen, fit_mins, fit_maxs, where=fit_maxs >= fit_mins, facecolor='g', alpha = 0.2)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.set_ylim([min(fit_mins)*0.98, max(fit_mins)*1.05])
    # ax1.legend(["Min", "Max", "Avg"], loc="Upper center")
    plt.grid(True)
    # plt.savefig("Convergencia.eps", dpi = 300)
plt.close('all')

plot_evolucion(0.8, 0.2, 1)

# %%

def extractInd(c,m,k):    
    fileName = 'Resultados\RES_SO_OOPNET\OOPNET_IND_' + str(c) + '_' + str(m) + '.txt'
    file     = open(fileName, 'r')
    data     = list(csv.reader(file, delimiter=","))
    file.close()
    ind0 = [ int(j.replace(']','').replace('[','')) for i,j in enumerate(data[k])][1:]
    ind  = [ [ind0[i+2],ind0[i+3]] for i in range(len(ind0)-2) if not i%2 ]
    ind.insert(0,0)
    return ind
        
def findBestFit(c,m):
    bestFit   = [100,0]
    fileName = 'Resultados\RES_SO_OOPNET\OOPNET_FIT_' + str(c) + '_' + str(m) + '.txt'
    file     = open(fileName, 'r')
    data     = list(csv.reader(file, delimiter=","))
    file.close()
    fitValues = [ float(j[4]) for i,j in enumerate(data)]
    bestFit    = min(fitValues)
    pos       = fitValues.index(bestFit)
    return bestFit, pos



# %%


from statistics import stdev, mean

def optPopStats():
    print('\n'*3)    
    print('-'*90)
    print(' c ','\t  m ','\t   Min','\t\t   Max','\t\t   Avg','\t\t   Std')
    print('-'*90)
    valores_c = [ 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2 ]
    valores_m = [ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ]
    for c,m in zip(valores_c, valores_m):         
        fileName = 'Resultados\RES_SO_OOPNET\OOPNET_FIT_' + str(c) + '_' + str(m) + '.txt'
        file     = open(fileName, 'r')
        data     = list(csv.reader(file, delimiter=","))
        file.close()
        
        bestFit, pos = findBestFit(c,m)
        ind = extractInd(c,m,pos)
        
        report = OOPNET.FitnessOOPNET(ind,depurar=True)
        
        # Q = report.pressure
        
        fitnessValues = [float(j[4]) for i,j in enumerate(data) if float(j[4])<10]
        fit_min = min(fitnessValues)
        fit_max = max(fitnessValues)
        fit_std = stdev(fitnessValues)
        fit_avg = mean(fitnessValues)
        
        print(c,'\t',m,'\t','{0:.6f}'.format(fit_min),
                  '\t','{0:.6f}'.format(fit_max),
                  '\t','{0:.6f}'.format(fit_avg),
                  '\t','{0:.6f}'.format(fit_std))
    print('-'*90)
    print('\n'*3)
    print('Mejor individuo: ', ind)
    
optPopStats()


# %%
    
def optIndStats():    
    print('-'*90)
    print(' c ','\t  m ','\t  L   ', '\t\t Qav', '\t\t Qsd')
    print('-'*90)
    valores_c = [ 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2 ]
    valores_m = [ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ]
    for c,m in zip(valores_c, valores_m):        
        bestFit, pos = findBestFit(c,m)
        ind = extractInd(c,m,pos)
        report = OOPNET.FitnessOOPNET(ind, depurar=True)
        Q = 1e3 * OOPNET.KE * report.pressure**0.5
        del Q['R']
        C = sum(report.length)
       
        print(c,'\t',m,'\t', '{0:.2f}'.format(C), '\t\t{0:.4f}'.format(mean(Q)), '\t\t{0:.4f}'.format(stdev(Q)) )
        
    print('-'*90)
    
optIndStats()

# %%


fit1, pos = findBestFit(0.5, 0.5)


ind0 = WNC.indRef[:]
ind1 = extractInd(0.5, 0.5, pos)

fit0     = OOPNET.FitnessOOPNET(ind0,depurar=False)[0]

report0 = OOPNET.FitnessOOPNET(ind0, depurar=True)
report1 = OOPNET.FitnessOOPNET(ind1, depurar=True)

C0 = sum(report0.length)
C1 = sum(report1.length)

P0 = report0.pressure
P1 = report1.pressure

Q0 = 1e3 * OOPNET.KE * report0.pressure**0.5
Q1 = 1e3 * OOPNET.KE * report1.pressure**0.5

del Q0['R']
del Q1['R']


print('\n'*5)
print('-'*50)
print(' '*5,' REFERENCE SOLUTION ')
print('-'*50)

print('Fitness ', fit0)
print('dQmax:  ', max(Q0)-min(Q0))
print('Qmean:  ', mean(Q0))
print('Qstdev: ', stdev(Q0))
print('Length: ', C0)

print('-'*50)
print(' '*5,' OPTIMAL SOLUTION ')
print('-'*50)

print('Fitness ', fit1)
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

V0 = sum(Q0*t0-5)
V1 = sum(Q1*t0-5)

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


# %% SACAR PRESIONES DEL CAUDAL

# Q = Kd * sqrt(H)
# H = (Q/Kd)**2

H0 = np.array([(q/(3.6e6*WNC.Kd))**2 for q in Q0])
H1 = np.array([(q/(3.6e6*WNC.Kd))**2 for q in Q1])
 
P0 = [ h/10 for h in H0]
P1 = [ h/10 for h in H1]

print('\nPresiones 0 (bar) : \n', P0, sep='\n')
print('\nPresiones 1 (bar): \n', P1, sep='\n')

# %%

from mpl_toolkits.axes_grid1 import make_axes_locatable

# Pressure in Pa, transformed into bar
X0m = np.reshape(X0[:49],[7,7])*1e5
X1m = np.reshape(X1[:49],[7,7])*1e5

X0m = np.transpose(X0m)
X1m = np.transpose(X1m)

X0m = np.flipud(X0m)
X1m = np.flipud(X1m)

# Create some data
data1 = np.reshape(X0m,[7,7])
data2 = np.reshape(X1m,[7,7])

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

minimo = min([np.min(X0m), np.min(X1m)])
maximo = max([np.max(X0m), np.max(X1m)])

# Plot data
im1 = ax1.imshow(data1, vmin=minimo, vmax=maximo, cmap='viridis')
im2 = ax2.imshow(data2, vmin=minimo, vmax=maximo, cmap='viridis')

# Create an axis on the right side of ax2 for the colorbar
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)

# Create colorbar
cbar = plt.colorbar(im2, cax=cax)

# Add colorbar to the left plot as well
cbar.ax.tick_params(labelsize=10)

for i in range(7):
   for j in range(7):
      c1 = X0m[j, i]*1e-6
      ax1.text(i, j, str(round(c1,2)), va='center', ha='center')
      c2 = X0m[j, i]*1e-6
      ax2.text(i, j, str(round(c2,2)), va='center', ha='center')      

plt.show()
