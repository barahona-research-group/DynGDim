#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 22:13:20 2020

@author: robert
"""


import networkx as nx
import sys as sys
import pickle as pickle
import yaml as yaml

from tqdm import tqdm
import epydemic as ep
import pylab as plt

import numpy as np
import os as os
import scipy as sc

from functools import partial
from multiprocessing import Pool


from dyngdim.dyngdim import run_local_dimension,run_single_source,run_all_sources
from dyngdim.io import save_local_dimensions
from dyngdim.dyngdim import initial_measure as delta_measure


def compute_SIR(beta):

    prob_removed_matrix = np.zeros((N, N))
    error = np.zeros((N, SIR_its-1))

    param[ep.SIR.P_INFECT] = beta

    for k in G.nodes:

        removed_matrix_results = np.zeros((SIR_its, N))

        for n in range(SIR_its):

            E.setUp(param)
            # dist = E._process.initialCompartmentDistribution()

            #set initial compartment distribution
            for i in G.nodes():
                if i != k:
                    E._process.changeCompartment(i, 'epydemic.SIR.S')
                else:
                    E._process.changeCompartment(i, 'epydemic.SIR.I')

            #run the model
            RUN = E.do(param)

            #calc prob_infected
            for i in G.nodes:
                if E._process._g.nodes[i]['compartment'] == 'epydemic.SIR.R':
                    removed_matrix_results[n][i] = 1

            #tear down model and rerun
            E.tearDown

        #save the results after SIR_its iterations
        prob_removed_matrix[k] = np.mean(removed_matrix_results, axis=0)

        #save the step-by-step error
        prob_removed_thoughav = np.cumsum(removed_matrix_results,axis=0)/np.outer(np.array(range(1, SIR_its+1)), np.ones(N))
        error_k = np.abs(np.array(prob_removed_thoughav)[:SIR_its-1] - np.array(prob_removed_thoughav)[1:SIR_its])#np.array(prob_removed_thoughav)[1:SIR_its]
        error_k[np.isnan(error_k)] = 0 #set nan values to 0
        
        error[k] = np.mean(error_k, axis = 1)

    return [prob_removed_matrix, error]



folder = "./outputs/fig_sir"
ps = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]



mu = 1.
beta_min = -1
beta_max = 1
beta_len = 21
n_processes = 4
SIR_its = 1000

t_min = -2
t_max = 2
n_t = 100
n_workers = 4
times = np.logspace(t_min, t_max, n_t)

for p in ps:
    print(p)
    G = nx.newman_watts_strogatz_graph(100,2,p)

    #G = nx.fast_gnp_random_graph(100,p)
    
    pos = nx.spring_layout(G)
    N = len(G.nodes)
    nx.draw(G)


    A = nx.adjacency_matrix(G)
    w,v = np.linalg.eigh(A.toarray())
    beta_crit = mu/np.max(w)
    print('Beta critical = ', beta_crit)
    beta_range = np.logspace(np.log10(beta_crit)+beta_min, np.log10(beta_crit)+beta_max, beta_len)


    # create a model and a dynamics to run it
    Model = ep.SIR()                      # the model (process) to simulate
    E = ep.StochasticDynamics(Model, G)   # use stochastic (Gillespie) dynamics

    param = dict()
    param[ep.SIR.P_REMOVE] = mu
    param[ep.SIR.P_INFECTED] = 0. #initial prob infected (override this and set initial infection distribution later)

    #compute results
    with Pool(processes = n_processes) as p_uc:  #initialise the parallel computation
        out = list(tqdm(p_uc.imap(compute_SIR, beta_range), total = beta_len))

    prob_removed_matrix = np.zeros([beta_len, N, N])
    error = np.zeros([beta_len, N, SIR_its-1])

    for i in range(beta_len):
        prob_removed_matrix[i] = out[i][0]
        error[i] = out[i][1]
        
    local_dimensions = run_local_dimension(G, times, n_workers=n_workers)
    
    pickle.dump([G, local_dimensions,prob_removed_matrix,error], open(folder + './sir_new_sims/prob_removed_matrix_ws_{}.pkl'.format(p),'wb'))

