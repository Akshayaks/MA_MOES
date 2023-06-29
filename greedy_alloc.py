import matplotlib.pyplot as plt
import numpy as np
import common
from ergodic_coverage import ErgCover
import jax.numpy as jnp
import ergodic_metric
import time
from utils import *
from explicit_allocation import scalarize_minmax
from BB_optimized import get_minimal_bounding_sphere, greedy_alloc

np.random.seed(101)
bounding_sphere = True
scalarize = False

""""
Just do a greedy allocation based on the amount of information on a map in a window of arbitary size centered on the agent
"""

if __name__ == "__main__":
    n_agents = 4
    n_scalar = 10
    cnt = 0
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

        start_pos = np.load("./start_positions/start_pos_ang_random_4_agents.npy",allow_pickle=True)
        problem.s0 = start_pos.item().get(file)

        incumbent = greedy_alloc(problem,n_agents)
        incumbent_erg = np.zeros(len(problem.pdfs))

        pdf = np.zeros((100,100))

        #Find the upper bound for the incumbent solution
        for k,v in incumbent.items():
            if len(v) > 1:
                if bounding_sphere:
                    # print("Computing MBS")
                    pdf_center, _ = get_minimal_bounding_sphere([problem.pdfs[a] for a in v],problem.nA,problem.pix)
                else:
                    pdf = np.zeros((100,100))
                    for a in v:
                        pdf += (1/len(v))*problem.pdfs[a]
            else:
                pdf = problem.pdfs[v[0]]

            # if not Bounding_sphere:
            pdf = np.asarray(pdf.flatten())
            
            # Just run ergodicity optimization for fixed iterations to get ergodic trajectory
            if bounding_sphere and len(v) > 1:
                control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True,direct_FC=pdf_center)
            else: 
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

        print("Files: ", file)
        print("\nBest allocation is: ", best_alloc)
        print("\nBest Individual ergodicity: ", incumbent_erg)

        run_times[file] = time.time() - start_time
        best_allocs[file] = best_alloc
        indv_erg_best[file] = incumbent_erg

        np.save("greedy_4_agents_runtime.npy", run_times)
        np.save("greedy_4_agents_best_alloc.npy", best_allocs)
        np.save("greedy_4_agents_indv_erg.npy", indv_erg_best)
