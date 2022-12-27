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

dx = [0,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0,-0.01,-0.02,-0.03,-0.04,-0.05,-0.06,-0.07,-0.08,-0.09,-0.1]
dy = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

traj_perturb = []
start_perturb = []
tj_og = []

for i in range(len(problem.pdfs)):
	if i == 1:
		break
	for j in range(n_agents):
		if j == 0:
			continue

		for k in range(len(dx)):
			print("dx,dy: ", dx,dy)
			start_pos = np.array([problem.s0[0]+3*dx[k],problem.s0[1],0])
			start_perturb.append(np.sqrt(dx[k]*dx[k]+dy[k]*dy[k]))
			print("perturnbed start")

			pdf = jnp.asarray(problem.pdfs[i].flatten())
			EC = ergodic_metric.ErgCalc(pdf,1,problem.nA,n_scalar,problem.pix)
			print("init ergcalc")
			control, erg, iters = ErgCover(pdf, 1, problem.nA, start_pos, n_scalar, problem.pix, 500, False, None, stop_eps=-1,grad_criterion=True)
			print("ran ergcover")
			_, tj = ergodic_metric.GetTrajXY(control, problem.s0[j*3:j*3+3])
			print("got traj")
			if len(traj_perturb) == 0:
				tj_og = tj
				traj_perturb.append(0)
			else:
				print("in elseee")
				print(len(tj_og),len(tj))
				traj_perturb.append(np.linalg.norm(tj-tj_og))

			print("start_perturb, traj_perturb: ", start_perturb[k],traj_perturb[k])
			# pdb.set_trace()

			# display_map(problem,problem.s0,tj=tj)

# 		### Find out when the wiggle starts ###
# 		avg = 0
# 		sum_e = 0
# 		prev_avg = -1
# 		iter_wiggle = 0 
# 		for idx,e in enumerate(erg):
# 			sum_e += e
# 			if idx % 100 == 0:
# 				prev_avg = avg
# 				avg = sum_e/100
# 				print(prev_avg,avg)
# 				if abs(prev_avg - avg) < 0.0005:
# 					print("Average not changing! Wiggle started")
# 					print("Wiggle at erg: ", e)
# 					print("Iteration number: ", idx)
# 					iter_wiggle = idx
# 					break
# 				sum_e = 0
# 		min_erg.append(min(erg))
# 		max_erg.append(max(erg))
# 		print("Wiggle bound: ", min_erg,max_erg)

# fig, ax = plt.subplots()

# ax.axhspan(min_erg[0], max_erg[0], facecolor='green', alpha=0.5)
# ax.axhspan(min_erg[1], max_erg[1], facecolor='red', alpha=0.5)
# # ax.axhspan(min_erg[2], max_erg[2], facecolor='red', alpha=0.5)

# plt.show()

# EC = ergodic_metric.ErgCalc(pdf,1,nA,n_scalar,problem.pix)
# erg_t = []   #Ergodicity of the best trajectory with time
# for j in np.arange(1,len(control),1):
# 	print("Length of trajectory: ", j)
# 	e = EC.fourier_ergodic_loss(control[0:j], problem.s0[0:3], False)
# 	erg_t.append(e)

# time = np.arange(1,problem.nA,1)
print("start_perturb: ", dx[1:])
print("traj_perturb: ", traj_perturb[1:])
plt.plot([x*3 for x in dx[1:]],traj_perturb[1:])
plt.title("Variation of trajectory norm with start_perturbation")
plt.xlabel("Start perturbation")
plt.ylabel("Trajectory norm change")
plt.show()

# print("Final Erg: ",erg_t[-1],erg[-1])