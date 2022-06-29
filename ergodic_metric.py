
import numpy as onp
import sys

from jax import vmap, jit, grad
import jax.numpy as np
from jax.lax import scan
from functools import partial
import pdb
from sklearn.preprocessing import normalize


def fDyn(x, u): # dynamics of the robot - point mass
	xnew = x + np.array([np.tanh(u[0]),np.tanh(u[0]),10*u[1]])
	return xnew, x

def fDiffDrive(x0, u):
	"""
	x0 = (x,y,theta)
	u = (v,w)
	"""
	# x = x0 + np.array([np.cos(x0[2])*np.abs(u[0]), np.sin(x0[2])*np.abs(u[0]), u[1]])
	x = x0 + np.array([np.cos(x0[2])*np.abs(u[0]), np.sin(x0[2])*np.abs(u[0]), 10*u[1]])
	return x, x0

def get_hk(k): # normalizing factor for basis function
	_hk = (2. * k + onp.sin(2 * k))/(4. * k)
	_hk = _hk.at[onp.isnan(_hk)].set(1.)
	return onp.sqrt(onp.prod(_hk))

def fk(x, k): # basis function
    return np.prod(np.cos(x*k))

def GetTrajXY(u, x0):
	"""
	"""
	xf, tr0 = scan(fDiffDrive, x0, u)
	tr = tr0[:,0:2] # take the (x,y) part of all points
	return xf, tr


class ErgCalc(object):
	"""
	modified from Ian's Ergodic Coverage code base.
	"""
	def __init__(self, pdf, n_agents, nA, n_fourier, nPix):
		# print("Number of agents: ", n_agents)
		self.n_agents = n_agents
		self.nPix = nPix
		self.nA = nA
		# aux func
		self.fk_vmap = lambda _x, _k: vmap(fk, in_axes=(0,None))(_x, _k)

		# fourier indices
		k1, k2 = np.meshgrid(*[np.arange(0, n_fourier, step=1)]*2)
		k = np.stack([k1.ravel(), k2.ravel()]).T
		self.k = np.pi*k

		# lambda, the weights of different bands.
		self.lamk = (1.+np.linalg.norm(self.k/np.pi,axis=1)**2)**(-4./2.)

		# the normalization factor
		hk = []
		for ki in k:
		    hk.append(get_hk(ki))
		self.hk = np.array(hk)

		# compute phik
		if isinstance(nPix,int) == True:
			X,Y = np.meshgrid(*[np.linspace(0,1,num=self.nPix)]*2)
		else: #Using this when using a window around the agent and the window is not a square
			X,Y = np.meshgrid(np.linspace(0,1,num=self.nPix[0]),np.linspace(0,1,num=self.nPix[1]))
		_s = np.stack([X.ravel(), Y.ravel()]).T
		print("nPix: ", self.nPix)
		print("Shape of vmap: ",vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k).shape)
		phik = np.dot(vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k), pdf) #vmap(p)(_s)
		phik = phik/phik[0]
		self.phik = phik/self.hk		  

		# for reconstruction
		self.phik_recon = np.dot(self.phik, vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k)).reshape(X.shape)
		
		# to compute gradient func
		self.gradient = jit(grad(self.fourier_ergodic_loss))

		return

	def get_ck(self, tr):
		"""
		given a trajectory tr, compute fourier coeffient of its spatial statistics.
		k is the number of fourier coeffs.
		"""
		ck = np.mean(vmap(partial(self.fk_vmap, tr))(self.k), axis=1)
		ck = ck / self.hk
		return ck

	def fourier_ergodic_loss(self, u, x0, flag=False): #here call Get TrajXY and get_ck for each agent Can add nA here
		ck = 0
		trajectories = []


		##Uncomment to debug
		if flag == True:
			print("Number of agents in loss function: ", self.n_agents)
			print("Length of horizon: ", self.nA)
			# print("X0 value: ", x0)
			# print("Current u values: ", u.shape)
		# 	pdb.set_trace()

		for i in range(self.n_agents):
			u_i = u[i*self.nA:(i+1)*self.nA]
			x0_i = x0[i*3:i*3+3]
			xf, tr = GetTrajXY(u_i, x0_i)
			trajectories.append(tr)
			ck_i = self.get_ck(tr)
			ck += ck_i
		ck = ck / (self.n_agents)
		
		traj_cost = 0 
		for i in range(self.n_agents):
			traj_cost += np.mean((trajectories[i] - np.array([0.5,0.5]))**8)
		ergodicity = np.sum(self.lamk*np.square(self.phik - ck)) + 3e-2 * np.mean(u**2) + traj_cost
		return ergodicity


	def traj_stat(self, u, x0):
		"""
		"""
		xf, tr = GetTrajXY(u, x0)
		ck = self.get_ck(tr)
		X,Y = np.meshgrid(*[np.linspace(0,1,num=self.nPix)]*2)
		_s = np.stack([X.ravel(), Y.ravel()]).T
		pdf = np.dot(ck, vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k)).reshape(X.shape)
		return pdf
