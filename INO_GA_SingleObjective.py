# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:26:54 2023

@author: atapi
"""

from deap  import base
from deap  import creator
from deap  import tools
from deap  import algorithms

import pandas                    as pd
import numpy                     as np
import matplotlib.pyplot         as plt
import INO_GeneticOperators      as GenOps
import OOPNET_fun                as OOPNET



#%% SINGLE OBJECTIVE MU-PLUS-LAMBDA GENETIC ALGORITHM

creator.create("ProblemSO", base.Fitness, weights=(-1,))
creator.create("individual", list, fitness=creator.ProblemSO)

toolbox = base.Toolbox()

toolbox.register("individual",      tools.initIterate, creator.individual, GenOps.G_IndividualGeneration)
toolbox.register("ini_poblacion",   tools.initRepeat, list, toolbox.individual)
toolbox.register("select",          tools.selTournament, tournsize = 3)

toolbox.register("evaluate",        OOPNET.FitnessOOPNET)
toolbox.register("mate",            GenOps.G_Crossover)
toolbox.register("mutate",          GenOps.G_Mutation)

# %% FUNCIONES

def launch_SO(CXPB, MUTPB):
    """ Los parámetros de entrada son la probabilidad de cruce y la
    probabilidad de mutación """

    NGEN   = 100
    MU     = 500
    LAMBDA = 500
    
    pop = toolbox.ini_poblacion(n=MU)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    logbook = tools.Logbook()
    
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB,
                                              MUTPB, NGEN, stats=stats,
                                              halloffame=hof, verbose=True)
    
    return pop, hof, logbook,


if __name__ == "__main__":
    
    # import multiprocessing
    # pool = multiprocessing.Pool(processes=4)

    # toolbox.register("map", pool.map)
    
    # Número de simulaciones
    Nsim = 1
    
    # Probabilidades (pcx, pmut)
    valores_m = [ 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2 ]
    valores_c = [ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ]
    
    # Para cada combinación de c y m
    for c,m in zip(valores_c, valores_m): 
        
        # Abrimos dos archivos de texto para almacenar los resultados              
        res_individuos = open('Resultados/RES_SO' + '/OOPNET_IND_' + str(c) + '_' + str(m) + '.txt', "a")
        res_fitness    = open('Resultados/RES_SO' + '/OOPNET_FIT_' + str(c) + '_' + str(m) + '.txt', "a")
        
        # Lanzamos N veces el algoritmo
        for i in range(Nsim): 
    
              print('\n\n'': Probabilidad ',c,'-',m,'. Iteración ',i+1,'/',Nsim,'\n\n')
              
              # Hacemos la llamada al algoritmo
              pop, hof, log = launch_SO(c, m)
              
              # Almacenamos el logbook en un csv independiente
              df_log = pd.DataFrame(log)
              log_filename = 'Resultados/RES_SO' + '/OOPNET_LOG_' + str(c) + '_' + str(m) + '_IT' + str(i+1) + '.csv'
              df_log.to_csv(log_filename, index=False)
            
              # Almacenamos la solución en los ficheros de texto
              for ide, ind in enumerate(pop):
                  
                  res_individuos.write(str(i))
                  res_individuos.write(",")
                  res_individuos.write(str(ide))
                  res_individuos.write(",")
                  res_individuos.write(str([j for i,j in enumerate(ind)]))
                  res_individuos.write("\n")
                  
                  res_fitness.write(str(i))
                  res_fitness.write(",")
                  res_fitness.write(str(ide))
                  res_fitness.write(",")
                  res_fitness.write(str(c))
                  res_fitness.write(",")
                  res_fitness.write(str(m))
                  res_fitness.write(",")
                  res_fitness.write(str(ind.fitness.values[0]))
                  res_fitness.write("\n")
          
        # Borramos la solución y cerramos los archivos
        del(pop)
        del(hof)
        res_fitness.close()
        res_individuos.close()

# %%




# %%



    
    
    

