import numpy as np
import matplotlib.pyplot as plt
import os
import common

r_BB = np.load("./Final_exp/BB_wt_scal_runtime.npy",allow_pickle=True)
r_km = np.load("./Final_exp/BB_k_means_MBS_runtime.npy",allow_pickle=True)
r_sp = np.load("./Final_exp/BB_sphere_MBS_runtime.npy",allow_pickle=True)
r_gr = np.load("./greedy_4_agents_runtime.npy",allow_pickle=True)
r_dist = np.load("./dist_4_agents_runtime.npy",allow_pickle=True)

a_BB = np.load("./Final_exp/BB_wt_scal_alloc.npy",allow_pickle=True)
a_km = np.load("./Final_exp/BB_k_means_MBS_alloc.npy",allow_pickle=True)
a_sp = np.load("./Final_exp/BB_sphere_MBS_alloc.npy",allow_pickle=True)
a_gr = np.load("./greedy_4_agents_best_alloc.npy",allow_pickle=True)
a_dist = np.load("./dist_4_agents_best_alloc.npy",allow_pickle=True)

e_BB = np.load("./Final_exp/BB_wt_scal_indv_erg.npy",allow_pickle=True)
e_km = np.load("./Final_exp/BB_k_means_MBS_erg.npy",allow_pickle=True)
e_sp = np.load("./Final_exp/BB_sphere_MBS_erg.npy",allow_pickle=True)
e_gr = np.load("./greedy_4_agents_indv_erg.npy",allow_pickle=True)
e_dist = np.load("./dist_4_agents_indv_erg.npy",allow_pickle=True)


r_BB = r_BB.ravel()[0]
r_km = r_km.ravel()[0]
r_sp = r_sp.ravel()[0]
r_gr = r_gr.ravel()[0]
r_dist = r_dist.ravel()[0]

a_BB = a_BB.ravel()[0]
a_km = a_km.ravel()[0]
a_sp = a_sp.ravel()[0]
a_gr = a_gr.ravel()[0]
a_dist = a_dist.ravel()[0]

e_BB = e_BB.ravel()[0]
e_km = e_km.ravel()[0]
e_sp = e_sp.ravel()[0]
e_gr = e_gr.ravel()[0]
e_dist = e_dist.ravel()[0]

n_maps = []
map_files = []
n_agents = 4
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
# breakpoint()


for pbm_file in os.listdir("./build_prob/random_maps/"):
    pbm_file_complete = "./build_prob/random_maps/" + pbm_file
    problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)

    if pbm_file not in r_BB.keys() or pbm_file not in r_km.keys(): # or pbm_file not in r_gr.keys() or pbm_file not in r_dist.keys():
        continue

    map_files.append(pbm_file)
    n_maps.append(len(problem.pdfs))
    data["Testno"].append(c)
    c += 1
    data["Nummaps"] = len(problem.pdfs)

map_files = [x for _, x in sorted(zip(n_maps, map_files))]
n_maps.sort()
count_maps = np.zeros(11)
for i in range(len(map_files)):
    pbm_file_complete = "./build_prob/random_maps/" + map_files[i]
    problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)
    print(len(problem.pdfs))
    count_maps[len(problem.pdfs)] += 1
print("/n n_maps: ", n_maps)
for k in range(11):
    print(k,count_maps[k])
    print("\n")
breakpoint()

# result = data.groupby(["Testno"])['Nummaps'].aggregate(np.median).reset_index()
# norm = plt.Normalize(result["Nummaps"].values.min(), result["Nummaps"].values.max())
# colors = plt.cm.copper_r(norm(result["Nummaps"]))

# for k in r_km.keys():
#     # print("\nRuntime BB: ", r_BB[k])
#     print("\nRuntime sim: ", r_km[k])
#     print("\n______")
#     # print("\nBest Alloc BB: ", a_BB[k])
#     print("\nBest alloc sim: ", a_km[k])
#     print("\n_______")
#     # print("\nIndv Erg BB: ", e_BB[k])
#     print("\nIndv Erg sim: ", e_km[k])
#     breakpoint()

# print("\n***************************")
# breakpoint()

runtime_BB = []
runtime_km = []
runtime_sp = []
runtime_gr = []
runtime_dist = []

indv_erg_diff = 0
incomplete_alloc = 0
r_imprv_perc = 0
n_km_better = 0
same_clustering = 0
same_clustering_sp = 0
count = 0
max_indv_erg = 0
n_maps_new = []
minmax_erg_diff = []

n_gr_better = 0
n_dist_better = 0
n_sp_better = 0
minmax_erg_diff_gr = []
minmax_erg_diff_dist = []
minmax_erg_diff_sp = []
indv_erg_diff_gr = 0
indv_erg_diff_dist = 0
r_imprv_sp = 0
r_imprv_gr = 0
r_imprv_dist = 0
too_big_diff = 0
km_time = 0
gr_time = 0
dist_time = 0
sp_time = 0

print("\nLen of map files: ", len(map_files))
print("\nLen of r_BB: ", len(r_BB))
print("\nLen of r_km: ", len(r_km))
print("\nLen of r_sp: ", len(r_sp))
print("\n Len of r_gr: ", len(r_gr))
print("\nLen of r_dist: ", len(r_dist))
breakpoint()

for j in range(len(map_files)):
    if j == 70:
        break
    k = map_files[j]
    pbm_file_complete = "./build_prob/random_maps/" + k
    problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)
    print(len(problem.pdfs))
    if k not in r_BB.keys() or k not in r_km.keys(): # or k not in r_gr.keys() or k not in r_dist.keys():
        continue
    if r_BB[k] > 0:
        n_maps_new.append(n_maps[j])
        # print("File: ", k)
        
        # print("\nRuntime BB: ", r_BB[k])
        # print("\nRuntime Sim: ", r_km[k])
        # print("\n______")
        
        # print("\nBest Alloc BB: ", a_BB[k])
        # print("\nBest alloc Sim: ", a_km[k])
        # print("\n_______")
       
        # print("\nIndv Erg BB: ", e_BB[k])
        # print("\nIndv Erg Sim: ", e_km[k])
        # breakpoint()

        if r_BB[k] > r_km[k]:
            n_km_better += 1
            r_imprv_perc += (r_BB[k] - r_km[k])/r_BB[k]
            km_time += r_km[k]
        
        if r_BB[k] > r_sp[k]:
            n_sp_better += 1
            r_imprv_sp += (r_BB[k] - r_sp[k])/r_BB[k]
            sp_time += r_sp[k]
        
        if r_BB[k] > r_gr[k]:
            n_gr_better += 1
            r_imprv_gr += (r_BB[k] - r_gr[k])/r_BB[k]
            gr_time += r_gr[k]
        
        if r_BB[k] > r_dist[k]:
            n_dist_better += 1
            r_imprv_dist += (r_BB[k] - r_dist[k])/r_BB[k]
            dist_time += r_dist[k]

        runtime_BB.append(r_BB[k])
        runtime_km.append(r_km[k])
        runtime_gr.append(r_gr[k])
        runtime_dist.append(r_dist[k])
        runtime_sp.append(r_sp[k])

        if len(e_BB[k]) < n_agents or len(e_km[k]) < n_agents:
            incomplete_alloc += 1
            print("***Incomplete Allocation****")
            print("e_BB: ", e_BB[k])
            print("e_km: ", e_km[k])
            breakpoint()
            continue

        # if max(e_BB[k]) > max(e_km[k]):
        #     print("File: ", k)
        #     print("BB alloc: ", a_BB[k])
        #     print("km alloc: ", a_km[k])
        #     print(max(e_BB[k]) - max(e_km[k]))
        #     breakpoint()
        minmax_erg_diff.append((-max(e_BB[k]) + max(e_km[k]))/max(e_BB[k]))
        minmax_erg_diff_sp.append((-max(e_BB[k]) + max(e_sp[k]))/max(e_BB[k]))
        minmax_erg_diff_gr.append((-max(e_BB[k]) + max(e_gr[k]))/max(e_BB[k]))
        minmax_erg_diff_dist.append((-max(e_BB[k]) + max(e_dist[k]))/max(e_BB[k]))
        # if(abs(max(e_BB[k]) - max(e_gr[k])) > 5):
        #     print("file: ", k)
        #     breakpoint()
        # print("\n Minmax metrics: \n")
        # print("\nBB: ", max(e_BB[k]))
        # print("\nBB_km: ", max(e_km[k]))
        # print("\nBB_sp: ", max(e_sp[k]))
        # print("\nGreedy: ", max(e_gr[k]))
        # print("\nDist: ", max(e_dist[k]))
        # breakpoint()
        count += 1
        
        # indv_erg_diff /= n_agents

        # km_alloc = []
        # bb_alloc = []
        # sp_alloc = []

        # for i in range(n_agents):
        #     km_alloc.append(a_km[k][i])
        #     bb_alloc.append(list(a_BB[k][i]))
        #     sp_alloc.append(a_sp[k][i])

        # breakpoint()
        # km_alloc.sort()
        # bb_alloc.sort()
        # sp_alloc.sort()

        # print(km_alloc, bb_alloc, sp_alloc)
        
        # if km_alloc == bb_alloc:
        #     same_clustering += 1
        # if sp_alloc == bb_alloc:
        #     same_clustering_sp += 1
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
print("\nNumber of cases where km clustering helped: ", n_km_better)
print("\nNumber of cases where MBS clustering helped: ", n_sp_better)
print("\nAverage percentage improvement in runtime km clustering: ", r_imprv_perc/n_km_better)
print("\nAverage percentage improvement in runtime MBS clustering: ", r_imprv_sp/n_sp_better)
print("\nAverage percentage improvement in runtime greedy: ", r_imprv_gr/n_gr_better)
print("\nAverage percentage improvement in runtime distance: ", r_imprv_dist/n_dist_better)
print("\n Average runtime BB: ", sum(r_BB.values())/len(r_BB))
print("\n Average runtime BB with km clustering: ", km_time/n_km_better)
print("\n Average runtime BB with MBS clustering: ", sp_time/n_sp_better)
print("\n Average runtime Greedy: ", gr_time/n_gr_better)
print("\n Average runtime dist-based: ", dist_time/n_dist_better)
print("\nNumber of cases with same clustering for km: ", same_clustering)
print("\nNumber of cases with same clustering for MBS: ", same_clustering_sp)
print(n_km_better,n_gr_better,n_dist_better,n_sp_better)
# print("\nDifference in maximum individual ergodicity: ", max_indv_erg/count)
print("/nAverage, max and min in diiference in maximum individual ergodicity km clustering: ", sum(minmax_erg_diff)/len(minmax_erg_diff), max(minmax_erg_diff), min(minmax_erg_diff))
print("/nStandard deviation BB km: ", np.std(minmax_erg_diff))
print("/nAverage, max and min in diiference in maximum individual ergodicity MBS clustering: ", sum(minmax_erg_diff_sp)/len(minmax_erg_diff_sp), max(minmax_erg_diff_sp), min(minmax_erg_diff_sp))
print("/nStandard deviation BB MBS: ", np.std(minmax_erg_diff_sp))
print("/nAverage, max and min in diiference in maximum individual ergodicity greedy: ", sum(minmax_erg_diff_gr)/len(minmax_erg_diff_gr), max(minmax_erg_diff_gr), min(minmax_erg_diff_gr))
print("/nAverage, max and min in diiference in maximum individual ergodicity distance: ", sum(minmax_erg_diff_dist)/len(minmax_erg_diff_dist), max(minmax_erg_diff_dist), min(minmax_erg_diff_dist))
print("\nToo big diff: ", too_big_diff)
# print("\nCount: ", count)

plt.rcParams.update({'font.size': 20})
# bar1 = plt.bar(ind, runtime_BB, width, color='#FF5C5C')
# bar2 = plt.bar(ind+width, runtime_km, width, color='#00D100')
# bar3 = plt.bar(ind+width, runtime_gr, width, color='#2E2EFF')
# bar4 = plt.bar(ind+width, runtime_dist, width, color='black')

plt.plot(ind,runtime_BB,'-o',color='#FF5C5C')
plt.plot(ind,runtime_gr,'-o',color='#2E2EFF')
plt.plot(ind,runtime_dist,'-o',color='black')
plt.plot(ind,runtime_km,'-o',color='#00D100')
plt.plot(ind,runtime_sp,'-o',color='#00D1FF')

xcoords = [11,24,36,43,52,57]
for xc in xcoords:
    plt.axvline(x=xc,color="black",linestyle="--")

# plt.xticks(ind+3*width,ind)
plt.legend(["BB","Greedy Allocation","Distance based Allocation","BB_k_means_clustering","BB_MBS_clustering"],loc="upper left",fontsize=8) #"Greedy Allocation","Distance based Allocation",
plt.title("Runtime of BB against baseline approaches on different test cases with 4 agents")
plt.xlabel("Test case in order of increasing number of objectives")
plt.ylabel("Runtime (sec)")
plt.yscale("log")
plt.show()










