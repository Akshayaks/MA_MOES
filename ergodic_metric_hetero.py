import numpy as np
from jax import vmap, jit, grad
import jax.numpy as jnp
from jax.lax import scan
from functools import partial
import yaml

# rob_vel = 0.8
# def fDyn(x, u): # dynamics of the robot - point mass
# 	xnew = x + jnp.array([jnp.tanh(u[0]),jnp.tanh(u[0]),10*u[1]])
# 	# xnew = x + jnp.array([0.8,0.8,10*u[1]])
# 	return xnew, x

with open("agent_profile.yaml", "r") as yamlfile:
    agent_profile = yaml.load(yamlfile, Loader=yaml.FullLoader)

def fDiffDrive0(x0, u, max_speed = 0.25):
	"""
	x0 = (x,y,theta)
	u = (v,w)
	"""
	# print("Speed limit: ", max_speed)
	u = max_speed * jnp.tanh(u) #Limit the maximum velocity
	x = x0 + jnp.array([jnp.cos(x0[2])*jnp.abs(u[0]), jnp.sin(x0[2])*jnp.abs(u[0]), 10*u[1]])
	return x, x0

def fDiffDrive1(x0, u, max_speed = 1):
	"""
	x0 = (x,y,theta)
	u = (v,w)
	"""
	print("Speed limit: ", max_speed)
	u = max_speed * jnp.tanh(u) #Limit the maximum velocity
	x = x0 + jnp.array([jnp.cos(x0[2])*jnp.abs(u[0]), jnp.sin(x0[2])*jnp.abs(u[0]), 10*u[1]])
	return x, x0

def fDiffDrive2(x0, u, max_speed = 10):
	"""
	x0 = (x,y,theta)
	u = (v,w)
	"""
	print("Speed limit: ", max_speed)
	u = max_speed * jnp.tanh(u) #Limit the maximum velocity
	x = x0 + jnp.array([jnp.cos(x0[2])*jnp.abs(u[0]), jnp.sin(x0[2])*jnp.abs(u[0]), 10*u[1]])
	return x, x0

def get_hk(k): # normalizing factor for basis function
	_hk = jnp.array((2. * k + np.sin(2 * k))/(4. * k))
	_hk = _hk.at[np.isnan(_hk)].set(1.)	
	return np.sqrt(np.prod(_hk))

def fk(x, k): # basis function
    return jnp.prod(jnp.cos(x*k))

def GetTrajXY(u, x0, max_speed):
    """
    """
    # fdynamics = lambda x,u: fDiffDrive(x, u, max_speed)
    # xf, tr0 = scan(fdynamics, x0, u)
    if max_speed == 0.25:
        xf, tr0 = scan(fDiffDrive0, x0, u)
    elif max_speed == 1:
        xf, tr0 = scan(fDiffDrive1, x0, u)
    else: 
        xf, tr0 = scan(fDiffDrive2, x0, u)
    tr = tr0[:,0:2] # take the (x,y) part of all points
    return xf, tr

def GetTrajXYTheta(u,x0):
	xf, tr = scan(fDiffDrive0, x0, u)
	return xf, tr


class ErgCalc(object):
	"""
	modified from Ian's Ergodic Coverage code base.
	"""
	def __init__(self, pdf, n_agents, agent_type, nA, n_fourier, nPix):
		# print("Number of agents: ", n_agents)
		self.n_agents = n_agents
		# print("Agent types: ", agent_type)
		self.nPix = nPix
		self.nA = nA
		self.agent_type = agent_type
		# aux func
		self.fk_vmap = lambda _x, _k: vmap(fk, in_axes=(0,None))(_x, _k)

		# fourier indices
		k1, k2 = jnp.meshgrid(*[jnp.arange(0, n_fourier, step=1)]*2)
		k = jnp.stack([k1.ravel(), k2.ravel()]).T
		self.k = jnp.pi*k

		# lambda, the weights of different bands.
		self.lamk = (1.+jnp.linalg.norm(self.k/jnp.pi,axis=1)**2)**(-4./2.)

		# the normalization factor
		hk = []
		for ki in k:
			hk.append(get_hk(ki))
		self.hk = jnp.array(hk)

		# compute phik
		if isinstance(nPix,int) == True:
			X,Y = jnp.meshgrid(*[jnp.linspace(0,1,num=self.nPix)]*2)
		else: #Using this when using a window around the agent and the window is not a square
			X,Y = jnp.meshgrid(jnp.linspace(0,1,num=self.nPix[0]),jnp.linspace(0,1,num=self.nPix[1]))
		_s = jnp.stack([X.ravel(), Y.ravel()]).T
		# print("nPix: ", self.nPix)
		# print("Shape of vmap: ",vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k).shape)
		phik = jnp.dot(vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k), pdf) #vmap(p)(_s)
		phik = phik/phik[0]
		self.phik = phik/self.hk		  

		# for reconstruction
		self.phik_recon = jnp.dot(self.phik, vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k)).reshape(X.shape)
		
		# to compute gradient func
		self.gradient = jit(grad(self.fourier_ergodic_loss))

		return
	
	def get_recon(self, FC):
		X,Y = jnp.meshgrid(*[jnp.linspace(0,1,num=self.nPix)]*2)
		_s = jnp.stack([X.ravel(), Y.ravel()]).T
		return jnp.dot(FC, vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k)).reshape(X.shape)

	def get_ck(self, tr):
		"""
		given a trajectory tr, compute fourier coeffient of its spatial statistics.
		k is the number of fourier coeffs.
		"""
		ck = jnp.mean(vmap(partial(self.fk_vmap, tr))(self.k), axis=1)
		ck = ck / self.hk
		return ck

	def fourier_ergodic_loss(self, u, x0, flag=False): 
		ck = 0
		trajectories = []

		##To debug
		if flag == True:
			print("Number of agents in loss function: ", self.n_agents)
			print("Length of horizon: ", self.nA)
			print("X0 value: ", x0)
			print("Current u values: ", u.shape)

		for i in range(self.n_agents):
			max_speed = agent_profile["agent_type_speeds"][str(self.agent_type)]
			# print("Max speed: ", max_speed)
			u_i = u[i*self.nA:(i+1)*self.nA]
			x0_i = x0[i*3:i*3+3]
			xf, tr = GetTrajXY(u_i, x0_i,max_speed)
			trajectories.append(tr)
			ck_i = self.get_ck(tr)
			ck += ck_i
		ck = ck / (self.n_agents)
		
		traj_cost = 0 
		for i in range(self.n_agents):
			traj_cost += jnp.mean((trajectories[i] - jnp.array([0.5,0.5]))**8)
		ergodicity = jnp.sum(self.lamk*jnp.square(self.phik - ck)) + 3e-2 * jnp.mean(u**2) + traj_cost
		return ergodicity
	
	def fourier_ergodic_loss_traj(self,traj):
		ck = self.get_ck(traj)
		traj_cost = jnp.mean((traj - jnp.array([0.5,0.5]))**8)
		ergodicity = jnp.sum(self.lamk*jnp.square(self.phik - ck)) + traj_cost
		return ergodicity

	def traj_stat(self, u, x0):
		"""
		"""
		xf, tr = GetTrajXY(u, x0)
		ck = self.get_ck(tr)
		X,Y = jnp.meshgrid(*[jnp.linspace(0,1,num=self.nPix)]*2)
		_s = jnp.stack([X.ravel(), Y.ravel()]).T
		pdf = jnp.dot(ck, vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k)).reshape(X.shape)
		return pdf
