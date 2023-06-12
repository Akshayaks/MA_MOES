import matplotlib.pyplot as plt
import numpy as np
import common
from ergodic_coverage_single_integrator import ErgCover
import jax.numpy as jnp
import ergodic_metric_single_integrator as EM
from utils import *

def simple_EC():
	n_agents = 2
	n_scalar = 10

	pbm_file = "build_prob/test_cases/2_maps_example_0.pickle"

	problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)

	problem.nA = 100

	problem.s0 = np.array([0.1,0.1,0.6,0.7])

	print("Agent start positions allotted:", problem.s0)

	# display_map(problem,problem.s0)

	for i in range(len(problem.pdfs)):
		if i == 1:
			break
		for j in range(n_agents):
			pdf = jnp.asarray(problem.pdfs[i].flatten())
			# EC = EM.ErgCalc(pdf,1,problem.nA,n_scalar,problem.pix)
			control, erg, iters = ErgCover(pdf, 1, problem.nA, problem.s0[j*2:j*2+2], n_scalar, problem.pix, 1000, False, None, stop_eps=-1,grad_criterion=True)
			print("Ergodicity achieved: ", erg[-1])
			# print("Controls final: ", control)
			_, tj = EM.GetTrajXY(control, problem.s0[j*2:j*2+2])
			print("Trajectory: ", tj)
			display_map(problem,problem.s0[i*2:i*2+2],{0:[0],1:[1]},tj=[tj,tj])
			# print("Length of traj and controls: ", len(tj),len(control))
			return

if __name__ == "__main__":
	simple_EC()
