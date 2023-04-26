import numpy as onp
import jax.numpy as np
from jax.experimental import optimizers

import matplotlib.pyplot as plt
import ergodic_metric
import pdb

GLOBAL_NUM_K = 0

def ErgCover(pdf, n_agents, nA, s0, n_fourier, nPix, nIter, ifDisplay, u_init=None, stop_eps=-1, kkk=0,grad_criterion=False,pdf1=None,pdf2=None,direct_FC=None):
	"""
	run ergodic coverage over a info map. Modified from Ian's code.
	return a list of control inputs.
	"""
	# print("****************************************************************")
	# print("[INFO] ErgCover, nA =", nA, " s0 =", s0, " n_fourier =", n_fourier, " stop_eps =", stop_eps)
	
	
	if direct_FC is not None:
		erg_calc = ergodic_metric.ErgCalc(pdf, n_agents, nA, n_fourier, nPix)
		erg_calc.phik = direct_FC
	else:
		print("Getting ErgCalc done")
		print("x0: ", s0)
		breakpoint()
		erg_calc = ergodic_metric.ErgCalc(pdf, n_agents, nA, n_fourier, nPix)
	if pdf1 is not None:
		erg_calc1 = ergodic_metric.ErgCalc(pdf1, n_agents, nA, n_fourier, nPix)
	if pdf2 is not None:
		erg_calc2 = ergodic_metric.ErgCalc(pdf2, n_agents, nA, n_fourier, nPix)

	opt_init, opt_update, get_params = optimizers.adam(1e-3) #Declaring Adam's optimizer

	# initial conditions
	x0 = np.array(s0)
	
	# print("Initial state of the agents: ", x0)
	
	u = np.zeros((nA*n_agents,2))
	if u_init is not None:
		u = np.array(u_init)
	opt_state = opt_init(u)
	log = []
	log1 = []
	log2 = []

	if stop_eps > 0:
		nIter = int(1e5) # set a large number, stop until converge.

	if grad_criterion == True: # We want to iterate till we find a minima 
		nIter = int(1000) # Again set a very large number of iterations
		# print("**Grad criterion activated!**")

	i = 0
	for i in range(nIter):
		# print("****Iter: ",i)
		g = erg_calc.gradient(get_params(opt_state), x0)

		opt_state = opt_update(i, g, opt_state)
		u = get_params(opt_state)
		log.append(erg_calc.fourier_ergodic_loss(u, x0).copy())
		if pdf1 is not None:
			log1.append(erg_calc1.fourier_ergodic_loss(u, x0).copy())
			log2.append(erg_calc2.fourier_ergodic_loss(u, x0).copy())
		# print("Erg: ", log[-1])

		## check for convergence
		if grad_criterion: # at least 10 iterationss
			if -0.01 < np.linalg.norm(g) < 0.01:
				# print("Reached grad criterion at iteration: ", i)
				# pdb.set_trace()
				break

		elif i > 10 and stop_eps > 0: # at least 10 iterationss
			if onp.abs(log[-1]) < stop_eps:
				# print("Reached final threshold before number of iterations!")
				break
		 

	if ifDisplay : # final traj
		plt.figure(figsize=(5,5))
		xf, tr = ergodic_metric.GetTrajXY(u, x0)
		X,Y = np.meshgrid(*[np.linspace(0,1,num=nPix)]*2)
		plt.contourf(X, Y, erg_calc.phik_recon, levels=np.linspace(np.min(erg_calc.phik_recon), np.max(erg_calc.phik_recon),100), cmap='gray')
		# plt.scatter(tr[:,0],tr[:,1], c='r', marker="*:")
		plt.plot(tr[0,0],tr[0,1], "ro:")
		plt.plot(tr[:,0],tr[:,1], "r.:")
		plt.axis("off")
		plt.pause(1)
		plt.savefig("build/plot_traj/MOES-O2-nA_"+str(nA)+"_num_"+str(kkk)+".png", bbox_inches='tight',dpi=200)
	if pdf1 is not None:
		return get_params(opt_state), log, i, log1, log2
	return get_params(opt_state), log, i

