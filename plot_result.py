import numpy as np
import matplotlib.pyplot as plt
import pdb
import os

x = np.arange(1,15)

runtime_BB = [50.98, 510.79, 338.12, 106.79, 193.86, 250.84, 328.50, 45.49, 443.24, 99.11, 258.58, 51.62, 232.47, 45.22]

runtime_MOES = [54.03, 508.73, 507.51, 112.96, 250.95, 262.97, 540.50, 51.47, 542.44, 117.68, 261.07, 52.36, 261.53, 50.96]

runtime_BB = np.load("BB_random_maps_runtime_2_agents.npy",allow_pickle=True)
runtime_BBi = np.load("BB_improved_random_maps_runtime_2_agents.npy",allow_pickle=True)
runtime_BB_EEE = np.load("BB_improved3_random_maps_runtime_2_agents.npy",allow_pickle=True)
runtime_MOES = np.load("MOES_random_maps_runtime_2_agents.npy",allow_pickle=True)

runtime_MOES = runtime_MOES.ravel()[0]
runtime_BB = runtime_BB.ravel()[0]
runtime_BBi = runtime_BBi.ravel()[0]
runtime_BB_EEE = runtime_BB_EEE.ravel()[0]

r_BB = []
r_MOES = []
r_BBi = []
r_BB_EEE = []
diff = []
higher_diff = []
maps = 0
cnt = 0
c2 = 0

for file in os.listdir("./build_prob/random_maps/"):
	if runtime_BB[file] > 0 and runtime_MOES[file] > 0:
		r_BB.append(runtime_BB[file])
		r_BBi.append(runtime_BBi[file])
		r_MOES.append(runtime_MOES[file])
		r_BB_EEE.append(runtime_BB_EEE[file])
		maps += 1
		# if runtime_MOES[file] < runtime_BBi[file]:
		print(runtime_MOES[file]-runtime_BBi[file])
		
		if runtime_BBi[file] > runtime_MOES[file]:
			cnt += 1
			higher_diff.append((runtime_BBi[file] - runtime_MOES[file])/runtime_MOES[file])
		else:
			c2 += 1
			diff.append((runtime_MOES[file] - runtime_BBi[file])/runtime_MOES[file])
	if maps == 50:
		break

print("\nMaps in which MOES took more time",c2)
print("\nAverage speed up: ", np.sum(diff)/len(diff))
print("Number of maps in which BB took more time: ", cnt)
print("\nAverge slower time: ", np.sum(higher_diff)/len(higher_diff))

# x = np.arange(1,len(r_MOES)+1)

# plt.bar(x + 0.2,r_MOES)
# plt.bar(x,r_BB,alpha=0.5)
# plt.bar(x - 0.2,r_BBi)
# plt.bar(x + 0.2,r_BB_EEE,alpha=0.5)
# plt.xticks(np.arange(len(x)), x)
# plt.legend(["MOES", "BB", "BB_improved","BB_EEE"])


N = 50
ind = np.arange(N)
width = 0.25

xvals = [8, 9, 2]
bar1 = plt.bar(ind, r_MOES, width, color = 'r')

yvals = [10, 20, 30]
bar2 = plt.bar(ind+width, r_BBi, width, color='g')

zvals = [11, 12, 13]
bar3 = plt.bar(ind+width*2, r_BB_EEE, width, color = 'b')

# plt.xlabel("Dates")
# plt.ylabel('Scores')
# plt.title("Players Score")
#   
plt.xticks(ind+width,ind)
# plt.legend( (bar1, bar2, bar3), ('Player1', 'Player2', 'Player3') )
# plt.show()
plt.legend(["MOES", "BB_random", "BB_EEE"])
plt.title("Runtime of MOES Vs BB on different maps with 2 agents")
plt.xlabel("Test case Number")
plt.ylabel("Runtime (sec)")
plt.show()