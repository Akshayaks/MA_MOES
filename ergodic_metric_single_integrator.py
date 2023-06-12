import numpy as np
from jax import vmap, jit, grad
import jax.numpy as jnp
from jax.lax import scan
from functools import partial

rob_vel = 0.8
def fDyn(x, u):
    # x: (x,y)
    # u: (vx,vy)
    # u = jnp.where(u > 0.5, 1, u)
    # u = jnp.where(u > 0.01, 1, u)
    # u = jnp.where(u < 0.01, 0, u)
    # u = jnp.where(u < -0.1, -1, u)
    # u = jnp.where(u > -0.1  u < 0.1, 0, u)
    xnew = x + jnp.array([u[0],u[1]])
    return xnew, x

def get_hk(k): # normalizing factor for basis function
	_hk = jnp.array((2. * k + np.sin(2 * k))/(4. * k))
	_hk = _hk.at[np.isnan(_hk)].set(1.)	
	return np.sqrt(np.prod(_hk))

def fk(x, k): # basis function
    return jnp.prod(jnp.cos(x*k))

def GetTrajXY(u, x0):
    """
    """
    xf, tr = scan(fDyn, x0, u)
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
			u_i = u[i*self.nA:(i+1)*self.nA]
			x0_i = x0[i*2:i*2+2]
			xf, tr = GetTrajXY(u_i, x0_i)
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
