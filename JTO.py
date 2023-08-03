import numpy as np
import common
from ergodic_coverage import ErgCover
import jax.numpy as jnp
import ergodic_metric
import time
from utils import *
from BB_optimized import get_minimal_bounding_sphere

bounding_sphere = False

""""
This files runs the Joint Trajectory Optimization on test cases. It considers the trajectory of all the agents as one long 
trajectory and optimizes the trajectory against the average of the information maps or against the MBS of the maps
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

        pdf = np.zeros((100,100))
        map_erg = np.zeros(len(problem.pdfs))

        if bounding_sphere:
            pdf_center, _ = get_minimal_bounding_sphere(problem.pdfs,problem.nA,problem.pix)
            pdf = np.asarray(pdf.flatten())
            control, erg, _ = ErgCover(pdf, n_agents, problem.nA, problem.s0, n_scalar, problem.pix, 1000, False, None, grad_criterion=True,direct_FC=pdf_center)
        else:
            for i in range(len(problem.pdfs)):
                pdf += (1/len(problem.pdfs))*problem.pdfs[i]
            pdf = np.asarray(pdf.flatten())
            control, erg, _ = ErgCover(pdf, n_agents, problem.nA, problem.s0, n_scalar, problem.pix, 1000, False, None, grad_criterion=True)

        #Find the upper bound for the incumbent solution
        for k in range(len(problem.pdfs)):
            pdf_indv = jnp.asarray(problem.pdfs[k].flatten())
            EC = ergodic_metric.ErgCalc(pdf_indv,4,problem.nA,n_scalar,problem.pix)
            map_erg[k] = EC.fourier_ergodic_loss(control,problem.s0)

        upper = max(map_erg)
        print("Incumber Ergodicities: ", map_erg)
        print("Initial Upper: ", upper)

        print("Files: ", file)
        print("\nBest Individual ergodicity: ", map_erg)

        run_times[file] = time.time() - start_time
        indv_erg_best[file] = map_erg

        np.save("JTO_4_agents_runtime.npy", run_times)
        np.save("JTO_4_agents_indv_erg.npy", indv_erg_best)