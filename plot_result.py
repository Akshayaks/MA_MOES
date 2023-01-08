import numpy as np
import matplotlib.pyplot as plt
import pdb
import os

x = np.arange(1,15)

runtime_BB = [50.98, 510.79, 338.12, 106.79, 193.86, 250.84, 328.50, 45.49, 443.24, 99.11, 258.58, 51.62, 232.47, 45.22]

runtime_MOES = [54.03, 508.73, 507.51, 112.96, 250.95, 262.97, 540.50, 51.47, 542.44, 117.68, 261.07, 52.36, 261.53, 50.96]

runtime_BB = np.load("BB_random_maps_runtime_2_agents.npy",allow_pickle=True)
runtime_BBi = np.load("BB_improved_random_maps_runtime_2_agents.npy",allow_pickle=True)
runtime_BB_EEE = np.load("BB_improved3_random_maps_runtime_4_agents.npy",allow_pickle=True)
runtime_MOES = np.load("MOES_random_maps_runtime_4_agents.npy",allow_pickle=True)
runtime_BB_similarity = np.load("BB_similarity_clustering_runtime_4_agents.npy",allow_pickle=True)

runtime_MOES = runtime_MOES.ravel()[0]
runtime_BB = runtime_BB.ravel()[0]
runtime_BBi = runtime_BBi.ravel()[0]
runtime_BB_EEE = runtime_BB_EEE.ravel()[0]
runtime_BB_similarity = runtime_BB_similarity.ravel()[0]

r_BB = []
r_MOES = []
r_BBi = []
r_BB_EEE = []
r_BB_sim = []
diff = []
higher_diff = []
maps = 0
cnt = 0
c2 = 0

# for file in os.listdir("./build_prob/random_maps/"):
# 	if file not in runtime_BB_EEE.keys():
# 		continue
# 	elif runtime_BB_EEE[file] > 0 and runtime_MOES[file] > 0 and runtime_BB_similarity[file] > 0:
# 		r_BB.append(runtime_BB[file])
# 		r_BBi.append(runtime_BBi[file])
# 		r_MOES.append(runtime_MOES[file])
# 		r_BB_EEE.append(runtime_BB_EEE[file])
# 		r_BB_sim.append(runtime_BB_similarity[file])
# 		maps += 1
# 		# if runtime_MOES[file] < runtime_BBi[file]:
# 		print(runtime_MOES[file]-runtime_BBi[file])
		
# 		if runtime_BBi[file] > runtime_MOES[file]:
# 			cnt += 1
# 			higher_diff.append((runtime_BBi[file] - runtime_MOES[file])/runtime_MOES[file])
# 		else:
# 			c2 += 1
# 			diff.append((runtime_MOES[file] - runtime_BBi[file])/runtime_MOES[file])
# 	if maps == 50:
# 		break

# print("\nMaps in which MOES took more time",c2)
# print("\nAverage speed up: ", np.sum(diff)/len(diff))
# print("Number of maps in which BB took more time: ", cnt)
# print("\nAverge slower time: ", np.sum(higher_diff)/len(higher_diff))

for k in runtime_BB_similarity.keys():
	if k not in runtime_BB_EEE.keys():
		continue
	if runtime_BB_EEE[k] <= 0 or runtime_BB_similarity[k] <= 0:
		continue
	r_BB_EEE.append(runtime_BB_EEE[k])
	r_BB_sim.append(runtime_BB_similarity[k])
	# r_MOES.append(runtime_MOES[k])

# x = np.arange(1,len(r_MOES)+1)

# plt.bar(x + 0.2,r_MOES)
# plt.bar(x,r_BB,alpha=0.5)
# plt.bar(x - 0.2,r_BBi)
# plt.bar(x + 0.2,r_BB_EEE,alpha=0.5)
# plt.xticks(np.arange(len(x)), x)
# plt.legend(["MOES", "BB", "BB_improved","BB_EEE"])
print("\nLength: ", len(r_BB_EEE), len(r_BB_sim), len(r_MOES))

N = len(r_BB_sim)
ind = np.arange(N)
width = 0.25


# bar1 = plt.bar(ind, r_MOES, width, color = 'r')


bar2 = plt.bar(ind, r_BB_EEE, width, color='g')


bar3 = plt.bar(ind+width, r_BB_sim, width, color = 'b')

# plt.xlabel("Dates")
# plt.ylabel('Scores')
# plt.title("Players Score")
#   
plt.xticks(ind+width,ind)
# plt.legend( (bar1, bar2, bar3), ('Player1', 'Player2', 'Player3') )
# plt.show()
plt.legend(["BB_EEE", "BB_sim"])
plt.title("Runtime of BB with and without clustering on different maps with 4 agents")
plt.xlabel("Test case Number")
plt.ylabel("Runtime (sec)")
plt.show()