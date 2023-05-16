import numpy as onp
import jax.numpy as np
from jax.experimental import optimizers

import matplotlib.pyplot as plt
import ergodic_metric_hetero
import yaml

with open("agent_profile.yaml", "r") as yamlfile:
    agent_profile = yaml.load(yamlfile, Loader=yaml.FullLoader)

GLOBAL_NUM_K = 0

def ErgCover(pdf, n_agents, nA, s0, agent_type, n_fourier, nPix, nIter, ifDisplay, u_init=None, stop_eps=-1, kkk=0,grad_criterion=False):
	"""
	run ergodic coverage over a info map. Modified from Ian's code.
	return a list of control inputs.
	"""
	# print("****************************************************************")
	# print("[INFO] ErgCover, nA =", nA, " s0 =", s0, " n_fourier =", n_fourier, " stop_eps =", stop_eps)
	
	erg_calc = ergodic_metric_hetero.ErgCalc(pdf, n_agents, agent_type, nA, n_fourier, nPix)

	opt_init, opt_update, get_params = optimizers.adam(1e-3) #Declaring Adam's optimizer

	# initial conditions
	x0 = np.array(s0)
	
	u = np.zeros((nA*n_agents,2))
	if u_init is not None:
		u = np.array(u_init)
	opt_state = opt_init(u)
	log = []

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
		max_speed = agent_profile["agent_type_speeds"][str(agent_type)]
		xf, tr = ergodic_metric_hetero.GetTrajXY(u, x0, max_speed)
		X,Y = np.meshgrid(*[np.linspace(0,1,num=nPix)]*2)
		plt.contourf(X, Y, erg_calc.phik_recon, levels=np.linspace(np.min(erg_calc.phik_recon), np.max(erg_calc.phik_recon),100), cmap='gray')
		# plt.scatter(tr[:,0],tr[:,1], c='r', marker="*:")
		plt.plot(tr[0,0],tr[0,1], "ro:")
		plt.plot(tr[:,0],tr[:,1], "r.:")
		plt.axis("off")
		plt.show()
		plt.pause(1)
		# plt.savefig("heterogeneous_agents"+str(nA)+"_num_"+str(kkk)+".png", bbox_inches='tight',dpi=200)
	return get_params(opt_state), log, i

