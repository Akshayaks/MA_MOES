import matplotlib.pyplot as plt
import numpy as np
import common
import scalarize
from ergodic_coverage import ErgCover
import jax.numpy as jnp
import pdb
from explicit_allocation import H_function
import ergodic_metric
from utils import *
import math

n_agents = 2
n_scalar = 10

pbm_file = "build_prob/test_cases/2_maps_example_0.pickle"

problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)

problem.nA = 10
nA = problem.nA

problem.s0 = np.array([0.1,0.1,0,0.6,0.7,0])
# random_start_pos(n_agents)

print("Agent start positions allotted:", problem.s0)

display_map(problem,problem.s0)

min_erg = []
max_erg = []

for i in range(len(problem.pdfs)):
	if i == 1:
		break
	for j in range(n_agents):
		print("Value of EEE function: ", H_function(problem.pdfs[i],problem.s0[j*3:j*3+3]))
		pdb.set_trace()

		pdf = jnp.asarray(problem.pdfs[i].flatten())
		EC = ergodic_metric.ErgCalc(pdf,1,problem.nA,n_scalar,problem.pix)

		control, erg, iters = ErgCover(pdf, 1, problem.nA, problem.s0[j*3:j*3+3], n_scalar, problem.pix, 500, False, None, stop_eps=-1,grad_criterion=True)
		time = np.arange(iters+1)
		plt.plot(time,erg)
		plt.title("Variation of ergodicity with number of iterations (for the entire length of the trajectory)")
		plt.xlabel("Iteration")
		plt.ylabel("Ergodicity of entire trajectory")
		plt.show()

		_, tj = ergodic_metric.GetTrajXY(control, problem.s0[j*3:j*3+3])
		print("Length of traj and controls: ", len(tj),len(control))
		print(control)
		pdb.set_trace()

		display_map(problem,problem.s0,tj=tj)

		### Find out when the wiggle starts ###
		avg = 0
		sum_e = 0
		prev_avg = -1
		iter_wiggle = 0 
		for idx,e in enumerate(erg):
			sum_e += e
			if idx % 100 == 0:
				prev_avg = avg
				avg = sum_e/100
				print(prev_avg,avg)
				if abs(prev_avg - avg) < 0.0005:
					print("Average not changing! Wiggle started")
					print("Wiggle at erg: ", e)
					print("Iteration number: ", idx)
					iter_wiggle = idx
					break
				sum_e = 0
		min_erg.append(min(erg))
		max_erg.append(max(erg))
		print("Wiggle bound: ", min_erg,max_erg)

fig, ax = plt.subplots()

ax.axhspan(min_erg[0], max_erg[0], facecolor='green', alpha=0.5)
ax.axhspan(min_erg[1], max_erg[1], facecolor='red', alpha=0.5)
# ax.axhspan(min_erg[2], max_erg[2], facecolor='red', alpha=0.5)

plt.show()

# EC = ergodic_metric.ErgCalc(pdf,1,nA,n_scalar,problem.pix)
# erg_t = []   #Ergodicity of the best trajectory with time
# for j in np.arange(1,len(control),1):
# 	print("Length of trajectory: ", j)
# 	e = EC.fourier_ergodic_loss(control[0:j], problem.s0[0:3], False)
# 	erg_t.append(e)

# time = np.arange(1,problem.nA,1)

# plt.plot(time,erg_t)
# plt.title("Variation of ergodicity with time")
# plt.xlabel("Time (t)")
# plt.ylabel("Ergodicity of trajectory [0:t]")
# plt.show()

# print("Final Erg: ",erg_t[-1],erg[-1])