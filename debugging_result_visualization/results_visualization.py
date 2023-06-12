import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import common

r_BB = np.load("./results_canebrake/BB_opt_runtime_3_agents.npy",allow_pickle=True)
r_BB2 = np.load("./results_canebrake/remaining_BB_opt_runtime_3_agents.npy",allow_pickle=True)
r_sim = np.load("./results_canebrake/BB_similarity_clustering_runtime_3_agents.npy",allow_pickle=True)
r_gr = np.load("./results_canebrake/greedy_3_agents_runtime.npy",allow_pickle=True)
r_dist = np.load("./results_canebrake/dist_3_agents_runtime.npy",allow_pickle=True)

a_BB = np.load("./results_canebrake/BB_opt_Best_alloc_3_agents.npy",allow_pickle=True)
a_BB2 = np.load("./results_canebrake/remaining_BB_opt_Best_alloc_3_agents.npy",allow_pickle=True)
a_sim = np.load("./results_canebrake/BB_similarity_clustering_best_alloc_3_agents.npy",allow_pickle=True)
a_gr = np.load("./results_canebrake/greedy_3_agents_best_alloc.npy",allow_pickle=True)
a_dist = np.load("./results_canebrake/dist_3_agents_best_alloc.npy",allow_pickle=True)

e_BB = np.load("./results_canebrake/BB_opt_indv_erg_3_agents.npy",allow_pickle=True)
e_BB2 = np.load("./results_canebrake/remaining_BB_opt_indv_erg_3_agents.npy",allow_pickle=True)
e_sim = np.load("./results_canebrake/BB_similarity_clustering_indv_erg_3_agents.npy",allow_pickle=True)
e_gr = np.load("./results_canebrake/greedy_3_agents_indv_erg.npy",allow_pickle=True)
e_dist = np.load("./results_canebrake/dist_3_agents_indv_erg.npy",allow_pickle=True)


r_BB = r_BB.ravel()[0]
r_BB2 = r_BB2.ravel()[0]
r_BB.update(r_BB2)
r_sim = r_sim.ravel()[0]
r_gr = r_gr.ravel()[0]
r_dist = r_dist.ravel()[0]

a_BB = a_BB.ravel()[0]
a_BB2 = a_BB2.ravel()[0]
a_BB.update(a_BB2)
a_sim = a_sim.ravel()[0]
a_gr = a_gr.ravel()[0]
a_dist = a_dist.ravel()[0]

e_BB = e_BB.ravel()[0]
e_BB2 = e_BB2.ravel()[0]
e_BB.update(e_BB2)
e_sim = e_sim.ravel()[0]
e_gr = e_gr.ravel()[0]
e_dist = e_dist.ravel()[0]

n_maps = []
map_files = []
n_agents = 3
data = {}
data["Testno"] = []
data["Nummaps"] = []
c = 1

# c3 = 0

# for pbm_file in os.listdir("./build_prob/random_maps/"):
#     pbm_file_complete = "./build_prob/random_maps/" + pbm_file
#     problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)
#     print(len(problem.pdfs))
    # c3 += 1

    # if len(problem.pdfs) < 3:
    #     c3 += 1

# print("Number of tests with less than 3 maps: ", c3)
# pdb.set_trace()


for pbm_file in os.listdir("./build_prob/random_maps/"):
    pbm_file_complete = "./build_prob/random_maps/" + pbm_file
    problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)

    if pbm_file not in r_BB.keys() or pbm_file not in r_sim.keys() or pbm_file not in r_gr.keys() or pbm_file not in r_dist.keys():
        continue

    map_files.append(pbm_file)
    n_maps.append(len(problem.pdfs))
    data["Testno"].append(c)
    c += 1
    data["Nummaps"] = len(problem.pdfs)

map_files = [x for _, x in sorted(zip(n_maps, map_files))]
n_maps.sort()
for i in range(len(map_files)):
    pbm_file_complete = "./build_prob/random_maps/" + map_files[i]
    problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)
    print(len(problem.pdfs))
print("/n n_maps: ", n_maps)
pdb.set_trace()

# result = data.groupby(["Testno"])['Nummaps'].aggregate(np.median).reset_index()

# norm = plt.Normalize(result["Nummaps"].values.min(), result["Nummaps"].values.max())
# colors = plt.cm.copper_r(norm(result["Nummaps"]))



# for k in r_sim.keys():
#     # print("\nRuntime BB: ", r_BB[k])
#     print("\nRuntime sim: ", r_sim[k])
#     print("\n______")
#     # print("\nBest Alloc BB: ", a_BB[k])
#     print("\nBest alloc sim: ", a_sim[k])
#     print("\n_______")
#     # print("\nIndv Erg BB: ", e_BB[k])
#     print("\nIndv Erg sim: ", e_sim[k])
#     pdb.set_trace()

# print("\n***************************")
# pdb.set_trace()

runtime_BB = []
runtime_sim = []
runtime_gr = []
runtime_dist = []

indv_erg_diff = 0
incomplete_alloc = 0
r_imprv_perc = 0
n_sim_better = 0
same_clustering = 0
count = 0
max_indv_erg = 0
n_maps_new = []
minmax_erg_diff = []

n_gr_better = 0
n_dist_better = 0
minmax_erg_diff_gr = []
minmax_erg_diff_dist = []
indv_erg_diff_gr = 0
indv_erg_diff_dist = 0
r_imprv_gr = 0
r_imprv_dist = 0

print("\nLen of map files: ", len(map_files))
print("\nLen of r_BB: ", len(r_BB))
print("\nLen of r_sim: ", len(r_sim))
print("\n Len of r_gr: ", len(r_gr))
print("\nLen of r_dist: ", len(r_dist))
pdb.set_trace()

for j in range(len(map_files)):
    k = map_files[j]
    pbm_file_complete = "./build_prob/random_maps/" + k
    problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)
    print(len(problem.pdfs))
    if k not in r_BB.keys() or k not in r_sim.keys():
        continue
    if r_BB[k] > 0:
        n_maps_new.append(n_maps[j])
        # print("File: ", k)
        
        # print("\nRuntime BB: ", r_BB[k])
        # print("\nRuntime Sim: ", r_sim[k])
        # print("\n______")
        
        # print("\nBest Alloc BB: ", a_BB[k])
        # print("\nBest alloc Sim: ", a_sim[k])
        # print("\n_______")
       
        # print("\nIndv Erg BB: ", e_BB[k])
        # print("\nIndv Erg Sim: ", e_sim[k])
        # pdb.set_trace()

        if r_BB[k] > r_sim[k]:
            n_sim_better += 1
            r_imprv_perc += (r_BB[k] - r_sim[k])/r_BB[k]
        
        if r_BB[k] > r_gr[k]:
            n_gr_better += 1
            r_imprv_gr += (r_BB[k] - r_gr[k])/r_BB[k]
        
        if r_BB[k] > r_dist[k]:
            n_dist_better += 1
            r_imprv_dist += (r_BB[k] - r_dist[k])/r_BB[k]

        runtime_BB.append(r_BB[k])
        runtime_sim.append(r_sim[k])
        runtime_gr.append(r_gr[k])
        runtime_dist.append(r_dist[k])

        if len(e_BB[k]) < n_agents or len(e_sim[k]) < n_agents:
            incomplete_alloc += 1
            print("***Incimplete Allocation****")
            print("e_BB: ", e_BB[k])
            print("e_sim: ", e_sim[k])
            # pdb.set_trace()
            continue

        minmax_erg_diff.append(abs(max(e_BB[k]) - max(e_sim[k]))/max(e_BB[k]))
        minmax_erg_diff_gr.append(abs(max(e_BB[k]) - max(e_gr[k]))/max(e_BB[k]))
        minmax_erg_diff_dist.append(abs(max(e_BB[k]) - max(e_dist[k]))/max(e_BB[k]))
        count += 1
        
        # indv_erg_diff /= n_agents

        sim_alloc = []
        bb_alloc = []

        for i in range(n_agents):
            sim_alloc.append(a_sim[k][i])
            bb_alloc.append(list(a_BB[k][i]))

        sim_alloc.sort()
        bb_alloc.sort()
        
        if sim_alloc == bb_alloc:
            same_clustering += 1
    else:
        print("\nRuntime of this case less than 0 for r_BB")


N = len(runtime_BB)
ind = np.arange(N)
width = 0.25

green_colors = ["#5CFF5C", "#00D100", "#00A300", "#007500"]
blue_colors = ["#00FFFF", "#0096FF", "#0000FF", "#00008B"]

cg = []
cb = []
k = 0

for i in range(len(n_maps_new)):
    if n_maps_new[i] < 6:
        k = 0
    elif n_maps_new[i] < 8:
        k = 1
    elif n_maps_new[i] < 10:
        k = 2
    else:
        k = 3

    cg.append(green_colors[k])
    cb.append(blue_colors[k])

print("\nNumber of experiments considered: ",N)
print("\nNumber of incomplete allocations: ", incomplete_alloc)
# print("\nAverage individual ergodicity difference: ", indv_erg_diff/(N-incomplete_alloc))
print("\nNumber of cases where clustering helped: ", n_sim_better)
print("\nAverage percentage improvement in runtime clustering: ", r_imprv_perc/n_sim_better)
print("\nAverage percentage improvement in runtime greedy: ", r_imprv_gr/n_gr_better)
print("\nAverage percentage improvement in runtime distance: ", r_imprv_dist/n_dist_better)
print("\nNumber of cases with same clustering: ", same_clustering)
# print("\nDifference in maximum individual ergodicity: ", max_indv_erg/count)
print("/nAverage, max and min in diiference in maximum individual ergodicity clustering: ", sum(minmax_erg_diff)/len(minmax_erg_diff), max(minmax_erg_diff), min(minmax_erg_diff))
print("/nAverage, max and min in diiference in maximum individual ergodicity greedy: ", sum(minmax_erg_diff_gr)/len(minmax_erg_diff_gr), max(minmax_erg_diff_gr), min(minmax_erg_diff_gr))
print("/nAverage, max and min in diiference in maximum individual ergodicity distance: ", sum(minmax_erg_diff_dist)/len(minmax_erg_diff_dist), max(minmax_erg_diff_dist), min(minmax_erg_diff_dist))
# print("\nCount: ", count)

plt.rcParams.update({'font.size': 25})
bar1 = plt.bar(ind, runtime_BB, width, color='red')
# bar2 = plt.bar(ind+width, runtime_sim, width, color='green')
# bar3 = plt.bar(ind+width, runtime_gr, width, color='blue')
bar4 = plt.bar(ind+width, runtime_dist, width, color='black')

# plt.xticks(ind+3*width,ind)
plt.legend(["BB", "Distance based allocation"]) #, "Greedy_Allocation","Distance_based_allocation"])
plt.title("Runtime of BB and Distance based allocation on different maps with 3 agents")
plt.xlabel("Test case")
plt.ylabel("Runtime (sec)")
plt.yscale("log")
plt.show()










