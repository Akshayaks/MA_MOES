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

n_agents = 1
n_scalar = 5

pbm_file = "build_prob/instances/MOES-O2-peaks_pix_100_baseline.pickle"

problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)

problem.nA = 100
nA = problem.nA

problem.s0 = random_start_pos(n_agents)

print("Agent start positions allotted:", problem.s0)

display_map(problem,problem.s0)

pdf = jnp.asarray(problem.pdfs[0].flatten())
EC = ergodic_metric.ErgCalc(pdf,1,problem.nA,n_scalar,problem.pix)

control, erg, iters = ErgCover(pdf, 1, problem.nA, problem.s0[0:3], n_scalar, problem.pix, 500, False, None, stop_eps=-1,grad_criterion=False)
time = np.arange(iters+1)
plt.plot(time,erg)
plt.title("Variation of ergodicity with number of iterations (for the entire length of the trajectory)")
plt.xlabel("Iteration")
plt.ylabel("Ergodicity of entire trajectory")
plt.show()

_, tj = ergodic_metric.GetTrajXY(control, problem.s0[:3])
print("Length of traj and controls: ", len(tj),len(control))

display_map(problem,problem.s0,tj=tj)

EC = ergodic_metric.ErgCalc(pdf,1,nA,n_scalar,problem.pix)
erg_t = []   #Ergodicity of the best trajectory with time
for j in np.arange(1,len(control),1):
	print("Length of trajectory: ", j)
	e = EC.fourier_ergodic_loss(control[0:j], problem.s0[0:3], False)
	erg_t.append(e)

time = np.arange(1,problem.nA,1)

plt.plot(time,erg_t)
plt.title("Variation of ergodicity with time")
plt.xlabel("Time (t)")
plt.ylabel("Ergodicity of trajectory [0:t]")
plt.show()

print("Final Erg: ",erg_t[-1],erg[-1])