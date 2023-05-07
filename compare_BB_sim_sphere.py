import numpy as np
import os
import common

if __name__ == "__main__":

    n_agents = 4

    best_alloc_bb = np.load("./results_canebrake/BB_opt_Best_alloc_4_agents.npy",allow_pickle=True)
    best_alloc_bb = best_alloc_bb.ravel()[0]

    best_alloc_sim = np.load("./Results_npy_files/BB_similarity_clustering_random_maps_best_alloc_4_agents.npy",allow_pickle=True)
    best_alloc_sim = best_alloc_sim.ravel()[0]

    best_alloc_sphere = np.load("./Best_clustering_minimal_bounding_sphere_more_than_7.npy",allow_pickle=True)
    best_alloc_sphere = best_alloc_sphere.ravel()[0]

    #####################################################

    indv_erg_bb = np.load("./results_canebrake/BB_opt_indv_erg_4_agents.npy",allow_pickle=True)
    indv_erg_bb = indv_erg_bb.ravel()[0]

    indv_erg_sim = np.load("./Results_npy_files/BB_similarity_clustering_random_maps_indv_erg_4_agents.npy",allow_pickle=True)
    indv_erg_sim = indv_erg_sim.ravel()[0]

    ## This has to be changed ##
    indv_erg_sphere = np.load("./Results_npy_files/BB_similarity_clustering_random_maps_indv_erg_4_agents.npy",allow_pickle=True)
    indv_erg_sphere = indv_erg_sphere.ravel()[0]

    ######################################################

    runtime_bb = np.load("./results_canebrake/BB_opt_random_maps_runtime_4_agents.npy",allow_pickle=True)
    runtime_bb = runtime_bb.ravel()[0]

    runtime_sim = np.load("./Results_npy_files/BB_similarity_clustering_random_maps_runtime_4_agents.npy",allow_pickle=True)
    runtime_sim = runtime_sim.ravel()[0]

    ## This has to be changed ##
    runtime_sphere = np.load("./Runtime_minimal_bounding_sphere_4_agents.npy",allow_pickle=True)
    runtime_sphere = runtime_sim

    count = 0

    same_alloc_sim = 0
    same_alloc_sphere = 0

    eq_alloc_sim = 0     #If the minmax metric of both the allocations is the same then equivalent allocation
    eq_alloc_sphere = 0

    no_runtime_sphere_greater = 0
    avg_inc_runtime = 0

    for file in os.listdir("build_prob/random_maps/"):
        
        pbm_file = "build_prob/random_maps/"+file
        if file not in best_alloc_bb.keys() or file not in best_alloc_sim.keys() or file not in best_alloc_sphere.keys():
            continue

        if file not in indv_erg_bb.keys() or file not in indv_erg_sim.keys() or file not in indv_erg_sphere.keys():
            continue

        if file not in runtime_bb.keys() or file not in runtime_sim.keys() or file not in runtime_sphere.keys():
            continue
        
        # print("\nFile: ", file)
        count += 1
        
        problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)

        matching_sim = True
        matching_sphere = True

        alloc_bb = best_alloc_bb[file]
        alloc_sim = best_alloc_sim[file]
        alloc_sphere = best_alloc_sphere[file]

        erg_bb = indv_erg_bb[file]
        erg_sim = indv_erg_sim[file]
        erg_sphere = indv_erg_sphere[file]

        # print("Alloc by BB opt: ", alloc_bb)
        # print("Alloc by BB sim: ", alloc_sim)
        # print("Alloc by BB sphere: ", alloc_sphere)

        for i in range(n_agents):
            if len(alloc_bb[i]) != len(alloc_sim[i]):
                # print("BB sim does not match")
                matching_sim = False
                break
            if not (np.array(alloc_bb[i]) == alloc_sim[i]).all():
                # print("BB sim does not match")
                matching_sim = False
                break

        if not matching_sim and max(erg_bb) == max(erg_sim):
            # print("Equivalent sim allocation")
            eq_alloc_sim += 1
                
        for i in range(n_agents):
            if len(alloc_bb[i]) != len(alloc_sphere[i]):
                # print("BB sphere does not match")
                matching_sphere = False
                break
            if not (np.array(alloc_bb[i]) == alloc_sphere[i]).all():
                # print("BB sphere does not match")
                matching_sphere = False
                break
        
        if not matching_sphere and max(erg_bb) == max(erg_sphere):
            # print("Equivalent sphere allocation")
            eq_alloc_sphere += 1
        
        if matching_sim:
            # print("Same BB sim")
            same_alloc_sim += 1
        
        if matching_sphere:
            # print("Same BB sphere")
            same_alloc_sphere += 1
        
        if runtime_sim[file] < runtime_sphere[file]:
            no_runtime_sphere_greater += 1
            avg_inc_runtime += (runtime_sphere[file] - runtime_sim[file])/runtime_sim[file]
    
    # print("Length of the output files: ", len(best_alloc_bb), len(best_alloc_sim), len(best_alloc_sphere))
    print("Total number of test cases considered: ", count)
    print("Number of cases when BB sim matches BB opt: ", same_alloc_sim)
    print("Number of cases when BB sphere matches BB opt: ", same_alloc_sphere)
    print("Number of cases when BB sim is equivalent to BB opt: ", eq_alloc_sim)
    print("Number of cases when BB sphere is equivalent to BB opt: ", eq_alloc_sphere)
    print("Number of cases when BB sphere has greater runtime than BB sim: ", no_runtime_sphere_greater)
    print("Average percentage increase in runtime for BB sphere compared to BB sim: ", avg_inc_runtime/no_runtime_sphere_greater)

        
