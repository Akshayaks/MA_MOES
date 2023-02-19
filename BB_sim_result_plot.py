import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import common
# import seaborn as sns


# r_BB = np.load("./results_canebrake/BB_improved3_random_maps_runtime_4_agents.npy",allow_pickle=True)
# r_sim = np.load("./results_canebrake/4to7maps_BB_similarity_FC_clustering_runtime_4_agents.npy",allow_pickle=True)
# r_ex = np.load("Exhaustive_random_maps_runtime_4_agents.npy",allow_pickle=True)

# a_BB = np.load("./results_canebrake/Best_alloc_BB_improved3_random_maps_4_agents.npy",allow_pickle=True)
# a_sim = np.load("./results_canebrake/4to7maps_BB_similarity_FC_clustering_best_alloc_4_agents.npy",allow_pickle=True)
# a_ex = np.load("Exhaustive_random_maps_best_alloc_4_agents.npy",allow_pickle=True)

# e_BB = np.load("./results_canebrake/BB_improved3_random_maps_indv_erg_4_agents.npy",allow_pickle=True)
# e_sim = np.load("./results_canebrake/4to7maps_BB_similarity_FC_clustering_indv_erg_4_agents.npy",allow_pickle=True)
# e_ex = np.load("Exhaustive_random_maps_indv_erg_4_agents.npy",allow_pickle=True)

r_BB = np.load("./results_canebrake/8to10_BB_improved3_random_maps_runtime_4_agents.npy",allow_pickle=True)
r_BB2 = np.load("./results_canebrake/new8to10_BB_improved3_random_maps_runtime_4_agents.npy",allow_pickle=True)
r_sim = np.load("./results_canebrake/8to10maps_BB_similarity_FC_clustering_runtime_4_agents.npy",allow_pickle=True)

a_BB = np.load("./results_canebrake/8to10_Best_alloc_BB_improved3_random_maps_4_agents.npy",allow_pickle=True)
a_BB2 = np.load("./results_canebrake/new8to10_Best_alloc_BB_improved3_random_maps_4_agents.npy",allow_pickle=True)
a_sim = np.load("./results_canebrake/8to10maps_BB_similarity_FC_clustering_best_alloc_4_agents.npy",allow_pickle=True)

e_BB = np.load("./results_canebrake/8to10_BB_improved3_random_maps_indv_erg_4_agents.npy",allow_pickle=True)
e_BB2 = np.load("./results_canebrake/new8to10_BB_improved3_random_maps_indv_erg_4_agents.npy",allow_pickle=True)
e_sim = np.load("./results_canebrake/8to10maps_BB_similarity_FC_clustering_indv_erg_4_agents.npy",allow_pickle=True)


r_BB = r_BB.ravel()[0]
r_BB2 = r_BB2.ravel()[0]
r_BB.update(r_BB2)
r_sim = r_sim.ravel()[0]

# r_ex = r_ex.ravel()[0]

a_BB = a_BB.ravel()[0]
a_BB2 = a_BB2.ravel()[0]
a_BB.update(a_BB2)
a_sim = a_sim.ravel()[0]
# a_ex = a_ex.ravel()[0]

e_BB = e_BB.ravel()[0]
e_BB2 = e_BB2.ravel()[0]
e_BB.update(e_BB2)
e_sim = e_sim.ravel()[0]
# e_ex = e_ex.ravel()[0]

n_maps = []
map_files = []
n_agents = 4
data = {}
data["Testno"] = []
data["Nummaps"] = []
c = 1

c3 = 0

for pbm_file in os.listdir("./build_prob/random_maps/"):
    pbm_file_complete = "./build_prob/random_maps/" + pbm_file
    problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)
    print(len(problem.pdfs))
    c3 += 1

    # if len(problem.pdfs) < 3:
    #     c3 += 1

print("Number of tests with less than 3 maps: ", c3)
pdb.set_trace()


for pbm_file in os.listdir("./build_prob/random_maps/"):
    pbm_file_complete = "./build_prob/random_maps/" + pbm_file
    problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)

    if len(problem.pdfs) < 8 or pbm_file not in r_BB.keys() or pbm_file not in r_sim.keys():
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

indv_erg_diff = 0
incomplete_alloc = 0
r_imprv_perc = 0
n_sim_better = 0
same_clustering = 0
count = 0
max_indv_erg = 0
n_maps_new = []

print("\nLen of map files: ", len(map_files))
print("\nLen of r_BB: ", len(r_BB))
print("\nLen of r_sim: ", len(r_sim))
pdb.set_trace()

for j in range(len(map_files)):
    k = map_files[j]
    pbm_file_complete = "./build_prob/random_maps/" + k
    problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)
    print(len(problem.pdfs))
    if k not in r_BB.keys():
        print("\nNot in BB results")
        if k in r_sim.keys():
            runtime_BB.append(0)
            runtime_sim.append(r_sim[k])
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
        
        else:
            print(a_BB[k])
            # continue

        runtime_BB.append(r_BB[k])
        runtime_sim.append(r_sim[k])

        # if r_BB[k] > r_sim[k]:
        #     n_sim_better += 1
        #     r_imprv_perc += (r_BB[k] - r_sim[k])/r_BB[k]
        
        # else:
        #     print(a_BB[k])
        #     pdb.set_trace()

        if len(e_BB[k]) < n_agents or len(e_sim[k]) < n_agents:
            incomplete_alloc += 1
            continue

        for i in range(n_agents):
            indv_erg_diff += abs(e_BB[k][i] - e_sim[k][i])
        
        max_indv_erg += abs(max(e_BB[k]) - max(e_sim[k]))
        count += 1
        
        indv_erg_diff /= n_agents

        sim_alloc = []
        bb_alloc = []

        for i in range(n_agents):
            sim_alloc.append(a_sim[k][i])
            bb_alloc.append(list(a_BB[k][i]))

        sim_alloc.sort()
        bb_alloc.sort()

        # print("sim_alloc: ", sim_alloc)
        # print("\nbb_alloc: ", bb_alloc)
        
        if sim_alloc == bb_alloc:
            # print("same")
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
    if n_maps_new[i] == 8:
        k = 0
    elif n_maps_new[i] == 9:
        k = 1
    elif n_maps_new[i] == 10:
        k = 2
    else:
        k = 3

    cg.append(green_colors[k])
    cb.append(blue_colors[k])

print("\nNumber of experiments considered: ",N)
print("\nAverage individual ergodicity difference: ", indv_erg_diff/(N-incomplete_alloc))
print("\nNumber of cases where clustering helped: ", n_sim_better)
print("\nAverage percentage improvement in runtime: ", r_imprv_perc/n_sim_better)
print("\nNumber of cases with same clustering: ", same_clustering)
print("\nDifference in maximum individual ergodicity: ", max_indv_erg/count)


bar1 = plt.bar(ind, runtime_BB, width, color=cg)
bar2 = plt.bar(ind+width, runtime_sim, width, color=cb)

plt.xticks(ind+width,ind)
plt.legend(["BB_random_fathoming", "BB_similarity_clustering"])
plt.title("Runtime of BB with and without clustering on different maps with 4 agents")
plt.xlabel("Test case Number")
plt.ylabel("Runtime (sec)")
plt.yscale("log")
plt.show()










