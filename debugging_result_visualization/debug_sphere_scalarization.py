import matplotlib.pyplot as plt
import numpy as np
import common
from ergodic_coverage import ErgCover
import ergodic_metric
from utils import *
from explicit_allocation import scalarize_minmax
from BB_optimized import get_minimal_bounding_sphere
import os

def simple_EC():
    n_agents = 4
    n_scalar = 10

    best_alloc = np.load("./results_canebrake/BB_opt_Best_alloc_4_agents.npy",allow_pickle=True)
    best_alloc = best_alloc.ravel()[0]

    best_alloc_sphere = np.load("./BB_opt_best_alloc_random_maps_4_agents_sphere.npy",allow_pickle=True)
    best_alloc_sphere = best_alloc_sphere.ravel()[0]

    for file in os.listdir("./build_prob/random_maps/"):
        print("File: ", file)
        pbm_file = "./build_prob/random_maps/"+file
        problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)
        problem.nA = 100
        pdf_list = problem.pdfs

        start_pos = np.load("./start_positions/start_pos_ang_random_4_agents.npy",allow_pickle=True)
        problem.s0 = start_pos.item().get(file)

        # print("Agent start positions allotted:", problem.s0)

        display_map_simple(problem,problem.s0)

        if "build_prob/random_maps/"+file not in best_alloc_sphere.keys():
            continue

        bb_alloc = best_alloc[file]
        sphere_alloc = best_alloc_sphere["build_prob/random_maps/"+file]

        print("Best allocation: ", bb_alloc)
        print("Best allocation sphere: ", sphere_alloc)

        if(len(bb_alloc) < n_agents+1 or len(sphere_alloc) < n_agents):
            continue
        breakpoint()

        allocs = [bb_alloc,sphere_alloc]
        c = 0

        for alloc in allocs:
            if c == 0:
                scalarize = False
                Bounding_sphere = False
            else:
                scalarize = False
                Bounding_sphere = True
            trajectories = [[] for k in range(len(problem.pdfs))]
            indv_erg = []
            pdf = np.zeros((100,100))

            for k,a in alloc.items():
                print("a: ", a)
                print("k: ", k)
                if a == []:
                    continue
                if len(a) > 1:
                    if scalarize:
                        pdf = scalarize_minmax([pdf_list[j] for j in a],problem.s0[k*3:k*3+3],problem.nA)
                    elif Bounding_sphere:
                        pdf_FC, _ = get_minimal_bounding_sphere([pdf_list[ai] for ai in a],problem.nA,problem.pix)
                        print("Got minimal bounding sphere")
                    else:
                        pdf = np.zeros((100,100))
                        for j in a:
                            pdf += (1/len(a))*pdf_list[j]
                else:
                    print("Single information map")
                    pdf = pdf_list[a[0]]

                pdf = np.asarray(pdf.flatten())

                #Just run ergodicity optimization for fixed iterations and see which agent achieves best ergodicity in that time
                if Bounding_sphere and len(a) > 1:
                    print("Optimizing trajectory against center of minimal bounding sphere")
                    control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True,direct_FC=pdf_FC)
                else:
                    control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)

                _, tj = ergodic_metric.GetTrajXY(control, problem.s0[k*3:k*3+3])
                print("Erg: ", erg[-1])
                # breakpoint()        

                #Can have a heuristic to show the order in which to evaluate the indv ergodicities
                for p in a:
                    trajectories[p] = tj
                    pdf_indv = np.asarray(pdf_list[p].flatten())
                    EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
                    erg = EC.fourier_ergodic_loss(control,problem.s0[3*k:3+3*k])
                    indv_erg.append(erg)
                
            display_map(problem,problem.s0,alloc,tj=trajectories)
            print("Individual ergodicities: ", indv_erg)
            print("Max indv erg: ", max(indv_erg))
            c += 1
            breakpoint()


if __name__ == "__main__":
	simple_EC()

