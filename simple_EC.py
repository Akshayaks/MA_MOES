import matplotlib.pyplot as plt
import numpy as np
import common
from ergodic_coverage import ErgCover
import jax.numpy as jnp
import ergodic_metric as EM
from utils import *
import copy
from explicit_allocation import main_run_comb_allocation

def simple_EC():
	n_agents = 2
	n_scalar = 10

	pbm_file = "build_prob/instances/MOES-O3-peaks_pix_100_simple.pickle"

	problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)

	problem.nA = 1000

	problem.s0 = np.array([0.5,0.8,0,0.6,0.4,0])

	print("Agent start positions allotted:", problem.s0)

	# display_map(problem,problem.s0)

	for i in range(len(problem.pdfs)):
		# if i == 1:
		# 	break
		# sum_v = 0
		# old = copy.copy(problem.pdfs[0])
		# for p in range(100):
		# 	for q in range(100):
		# 		old[p][q] = 1 - old[p][q]
		# 		sum_v += old[p][q]
		# old = (1/sum_v)*old
		for j in range(n_agents):
			pdf = jnp.asarray(problem.pdfs[i].flatten())
			# EC = EM.ErgCalc(pdf,1,problem.nA,n_scalar,problem.pix)
			# control, erg, iters = ErgCover(pdf, 1, problem.nA, problem.s0[j*3:j*3+3], n_scalar, problem.pix, 1000, False, None, stop_eps=-1,grad_criterion=True)
			# print("Ergodicity achieved: ", erg[-1])
			# print("Controls final: ", control)
			# breakpoint()
			# for c in range(len(control)):
			# 	control = control.at[c].set(np.random.rand(2)*control[c])	
				# control[c] = np.random.rand(2)*control[c]
			# _, tj = EM.GetTrajXY(control, problem.s0[j*3:j*3+3])
			# print("Trajectory: ", tj)
			display_map(problem,problem.s0,{0:[0,1,2],1:[0,1,2]}) #,tj=[tj,tj])
			# print("Length of traj and controls: ", len(tj),len(control))
			# return
			if j == 1:
				break

if __name__ == "__main__":
	simple_EC()
	# pbm_file = "MOES-O3-peaks_pix_100_simple.pickle"
	# main_run_comb_allocation(pbm_file,2)
