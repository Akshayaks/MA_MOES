import numpy as np
import os
import common

if __name__ == "__main__":

    n_agents = 4

    best_alloc_MBS = np.load("./Final_exp/BB_MBS_scal_alloc.npy",allow_pickle=True)
    best_alloc_MBS = best_alloc_MBS.ravel()[0]

    best_alloc_wt = np.load("./Final_exp/BB_wt_scal_alloc.npy",allow_pickle=True)
    best_alloc_wt = best_alloc_wt.ravel()[0]

    indv_erg_MBS = np.load("./Final_exp/BB_MBS_scal_indv_erg.npy",allow_pickle=True)
    indv_erg_MBS = indv_erg_MBS.ravel()[0]

    indv_erg_wt = np.load("./Final_exp/BB_wt_scal_indv_erg.npy",allow_pickle=True)
    indv_erg_wt = indv_erg_wt.ravel()[0]

    runtime_MBS = np.load("./Final_exp/BB_MBS_scal_runtime.npy",allow_pickle=True)
    runtime_MBS = runtime_MBS.ravel()[0]

    runtime_wt = np.load("./Final_exp/BB_wt_scal_runtime.npy",allow_pickle=True)
    runtime_wt = runtime_wt.ravel()[0]

    count = 0

    same_alloc = 0

    eq_alloc = 0     #If the minmax metric of both the allocations is the same then equivalent allocation

    no_runtime_sphere_greater = 0
    avg_inc_runtime = 0
    sphere_higher_minmax = 0

    for file in os.listdir("build_prob/random_maps/"):
        
        pbm_file = "build_prob/random_maps/"+file
        if file not in best_alloc_MBS.keys() or file not in best_alloc_wt.keys():
            continue

        if file not in indv_erg_MBS.keys() or file not in indv_erg_wt.keys():
            continue

        if file not in runtime_MBS.keys() or file not in runtime_wt.keys():
            continue
        
        # print("\nFile: ", file)
        count += 1
        
        problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)

        matching = True

        alloc_MBS = best_alloc_MBS[file]
        alloc_wt = best_alloc_wt[file]

        erg_MBS = indv_erg_MBS[file]
        erg_wt = indv_erg_wt[file]

        for i in range(n_agents):
            if len(alloc_MBS[i]) != len(alloc_wt[i]):
                # print("BB wt does not match")
                matching = False
                break
            if not (np.array(alloc_MBS[i]) == alloc_wt[i]).all():
                # print("BB wt does not match")
                matching = False
                break

        if not matching and max(erg_MBS) == max(erg_wt):
            # print("Equivalent wt allocation")
            eq_alloc += 1
        
        if matching:
            # print("Same BB wt")
            same_alloc += 1
        
        if runtime_wt[file] < runtime_MBS[file]:
            no_runtime_sphere_greater += 1
            avg_inc_runtime += (runtime_MBS[file] - runtime_wt[file])/runtime_wt[file]
        
        if max(erg_MBS) > max(erg_wt):
            sphere_higher_minmax += 1
            print("File: ", file)
            print("Diff: ", max(erg_MBS) - max(erg_wt))
            breakpoint()
    
    # print("Length of the output files: ", len(best_alloc_MBS), len(best_alloc_wt), len(best_alloc_sphere))
    print("Total number of test cases considered: ", count)
    print("Number of cases when BB wt matches BB MBS: ", same_alloc)
    print("Number of cases when BB wt is equivalent to BB MBS: ", eq_alloc)
    print("Number of cases where minmax of BB MBS is greater than BB wt: ", sphere_higher_minmax)
    print("Number of cases when BB MBS has greater runtime than BB wt: ", no_runtime_sphere_greater)
    print("Average percentage increase in runtime for BB MBS compared to BB wt: ", avg_inc_runtime/no_runtime_sphere_greater)

        
