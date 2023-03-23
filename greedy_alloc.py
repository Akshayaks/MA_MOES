import matplotlib.pyplot as plt
import numpy as np

import common
import scalarize
from ergodic_coverage import ErgCover
import jax.numpy as jnp
import pdb
import ergodic_metric

import time
from utils import *
from explicit_allocation import *
import json

np.random.seed(101)

""""
Just do a greedy allocation based on the amount of information on a map in a window of arbitary size centered on the agent
"""

def greedy_alloc(problem, n_agents,node = None, sensor_footprint = 15):
    #Allocate based on information in a window centered around the agent

    n_obj = len(problem.pdfs)
    print("Len objec: ", n_obj)
    agents_allotted = []
    maps_allotted = []
    if node:
        current_node = node
        while current_node.parent:
            agents_allotted.append(current_node.agent)
            maps_allotted = maps_allotted + list(current_node.tasks)
            current_node = current_node.parent

    if len(agents_allotted) == n_agents:
        print("all agents are allotted")
        return -1

    agent_locs = []
    for i in range(n_agents):
        if i in agents_allotted:
            agent_locs.append((-1,-1))
        else:
            agent_locs.append((round(problem.s0[0+i*3]*100),round(problem.s0[1+i*3]*100)))

    agent_scores = np.zeros((n_agents,n_obj))

    # Calculate how much information agent i gets when allotted map p

    for i in range(n_agents):
        if i in agents_allotted:
            continue

        x_min = max(agent_locs[i][0]-sensor_footprint,0)
        x_max = min(agent_locs[i][0]+sensor_footprint,100)

        y_min = max(agent_locs[i][1]-sensor_footprint,0)
        y_max = min(agent_locs[i][1]+sensor_footprint,100)

        for p in range(n_obj):
            if p in maps_allotted:
                continue
            agent_scores[i][p] = np.sum(problem.pdfs[p][y_min:y_max,x_min:x_max])

    allocation = {new_list: [] for new_list in range(n_agents)}

    agents_assigned = np.zeros(n_agents)
    print("agent scores: ", agent_scores)

    for i in range(n_obj):
        if i in maps_allotted:
            continue
        k = 0
        print("i: ", i)

        # Sort the amount of information on a map for different agents in descending order

        map_scores = [a[i] for a in agent_scores]
        info = sorted(map_scores, reverse=True) 
        found = False
        while not found:
            idx = map_scores.index(info[k]) # Free agent with max info on this map
            if (agents_assigned[idx] > 0 and np.count_nonzero(agents_assigned == 0) == n_obj - sum(agents_assigned)) or idx in agents_allotted:
                k += 1
            else:
                allocation[idx] = allocation[idx] + [i]
                found = True
                agents_assigned[idx] += 1
        
    print("Incumbent allocation: ", allocation)
    return allocation

if __name__ == "__main__":
    n_agents = 3
    n_scalar = 10
    cnt = 0

    file1 = open("greedy_allocate_3_agents.txt","w")
    run_times = {}
    best_allocs = {}
    indv_erg_best = {}

    for file in os.listdir("build_prob/random_maps/"):
        start_time = time.time()
        pbm_file = "build_prob/random_maps/"+file

        print("\nFile: ", file)

        problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)

        if len(problem.pdfs) < n_agents:
            continue

        start_pos = np.load("start_pos_random_3_agents.npy",allow_pickle=True)
        problem.s0 = start_pos.item().get(file)

        incumbent = greedy_alloc(problem,n_agents)

        incumbent_erg = np.zeros(len(problem.pdfs))

        #Find the upper bound for the incumbent solution
        for k,v in incumbent.items():
            if len(v) > 1:
                if scalarize:
                    pdf = scalarize_minmax([problem.pdfs[a] for a in v],problem.s0[k*3:k*3+3],problem.nA)
                else:
                    pdf = np.zeros((100,100))
                    for a in v:
                        pdf += (1/len(v))*problem.pdfs[a]
            else:
                pdf = problem.pdfs[v[0]]

            pdf = jnp.asarray(pdf.flatten())
            
            # Just run ergodicity optimization for fixed iterations to get ergodic trajectory 
            control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)
            
            # Calculate individual ergodicities using the gotten trajectory
            for p in v:
                pdf_indv = jnp.asarray(problem.pdfs[p].flatten())
                EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
                incumbent_erg[p] = EC.fourier_ergodic_loss(control,problem.s0[3*k:3+3*k])

        upper = max(incumbent_erg)
        print("Incumbent allocation: ", incumbent)
        print("Incumber Ergodicities: ", incumbent_erg)
        print("Initial Upper: ", upper)
        
        best_alloc = incumbent

        print("\nBest allocation is: ", best_alloc)
        print("\nBest Individual ergodicity: ", incumbent_erg)            
        # pdb.set_trace()

        run_times[file] = time.time() - start_time
        best_allocs[file] = best_alloc
        indv_erg_best[file] = incumbent_erg

        np.save("greedy_3_agents_runtime.npy", run_times)
        np.save("greedy_3_agents_best_alloc.npy", best_allocs)
        np.save("greedy_3_agents_indv_erg.npy", indv_erg_best)

        file1.write(file)
        file1.write("\n")
        file1.write(json.dumps(incumbent))
        file1.write("\n")

    file1.close()


