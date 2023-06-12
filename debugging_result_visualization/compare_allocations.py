import numpy as np
import matplotlib.pyplot as plt

best_alloc_bb = np.load("./results_canebrake/BB_opt_Best_alloc_4_agents.npy",allow_pickle=True)
indv_erg_bb = np.load("./results_canebrake/BB_opt_indv_erg_4_agents.npy",allow_pickle=True)
runtime_bb = np.load("./results_canebrake/BB_opt_runtime_4_agents.npy",allow_pickle=True)
best_alloc_bb = best_alloc_bb.ravel()[0]
indv_erg_bb = indv_erg_bb.ravel()[0]
runtime_bb = runtime_bb.ravel()[0]

best_alloc_sim = np.load("BB_similarity_clustering_random_maps_best_alloc_4_agents.npy",allow_pickle=True)
indv_erg_sim = np.load("BB_similarity_clustering_random_maps_indv_erg_4_agents.npy",allow_pickle=True)
runtime_sim = np.load("BB_similarity_clustering_random_maps_runtime_4_agents.npy",allow_pickle=True)
# best_alloc_sim = np.load("4to7maps_BB_similarity_clustering_best_alloc_4_agents.npy",allow_pickle=True)
# indv_erg_sim = np.load("4to7maps_BB_similarity_clustering_indv_erg_4_agents.npy",allow_pickle=True)
best_alloc_sim = best_alloc_sim.ravel()[0]
indv_erg_sim = indv_erg_sim.ravel()[0]
runtime_sim = runtime_sim.ravel()[0]

print("Len of bb alloc: ", len(best_alloc_bb))
print("Len of sim alloc: ", len(best_alloc_sim))
n_agents = 4
count = 0
same_alloc = 0
r_bb = []
r_sim = []

for k in best_alloc_sim.keys():
    if k not in best_alloc_bb.keys():
        continue
    
    alloc_bb = best_alloc_bb[k]
    alloc_sim = best_alloc_sim[k]

    if len(alloc_bb) < n_agents + 1 or len(alloc_sim) < n_agents + 1:
        print("Incomplete")
        continue
    count += 1
    r_bb.append(runtime_bb[k])
    r_sim.append(runtime_sim[k])

    print(alloc_bb,alloc_sim)
    matching = True

    for i in range(n_agents):
        if len(alloc_bb[i]) != len(alloc_sim[i]):
            print("Not same")
            print("BB min_max: ", max(indv_erg_bb[k]))
            print("Sim min_max: ", max(indv_erg_sim[k]))
            if max(indv_erg_bb[k]) == max(indv_erg_sim[k]):
                same_alloc += 1
            matching = False
            # breakpoint()
            break
        if not (np.array(alloc_bb[i])==alloc_sim[i]).all():
            print("Not same")
            print("BB min_max: ", max(indv_erg_bb[k]))
            print("Sim min_max: ", max(indv_erg_sim[k]))
            if max(indv_erg_bb[k]) == max(indv_erg_sim[k]):
                same_alloc += 1
            matching = False
            # breakpoint()
            break
    if matching:
        print("Same")
        same_alloc += 1
    # breakpoint()
print("Total number of tests: ", count)
print("Number of matching alloc: ", same_alloc)


tests = np.arange(0,len(r_bb))
plt.bar(tests,r_bb)
plt.bar(tests,r_sim)
plt.legend(["BB","BB_with_clustering"])
plt.show()

