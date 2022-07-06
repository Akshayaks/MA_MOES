import sys
import numpy as np

import jax.numpy as jnp
from jax import vmap
from ergodic_coverage import ErgCover
import ergodic_metric

import matplotlib.pyplot as plt
import time

np.random.seed(100)

def ScalarizeInfoMaps(info_maps, weights):
  """
  input info_maps can be either prob distribution or phik.
  Note: Arbitrary #obj.
  """
  if len(info_maps) != len(weights):
    sys.exit("[ERROR] ScalarizeInfoMaps, input info_maps and weights have diff len!")
  if abs(np.sum(weights)-1) > 1e-6:
    sys.exit("[ERROR] ScalarizeInfoMaps, input weights do not sum to one!")
  # print("Info map: ", info_maps)
  print("Shape of info_maps in scalarizeInfoMaps: ", len(info_maps),len(info_maps[0]))
  out = np.zeros(info_maps[0].shape)
  for i, info_map in enumerate(info_maps):
    out += info_map*weights[i]
  return out

def RunScalarizeMethodOnce(pbm, n_agents, w, n_basis, n_iter, ifDisplay, u_init=None, stop_eps=1e-2, kkk=0, comb=False):
  """
  w - a weight vector.
  pbm - the problem object.
  Note: Arbitrary #obj.
  """
  # print("Length of weight vector: ", len(w))
  if comb:
    w = w[0]
    pdfs = pbm.pdfs
  elif len(w) == 2:
    pdfs = [pbm.pdf1, pbm.pdf2]
  elif len(w) == 3:
    pdfs = [pbm.pdf1, pbm.pdf2, pbm.pdf3]
  elif len(w) > 3:
    pdfs = pbm.pdfs 
  weighted_pdf = ScalarizeInfoMaps(pdfs, w)

  x = np.linspace(0,100,num=100)
  y = np.linspace(0,100,num=100)
  X,Y = np.meshgrid(x,y)

  pdf = jnp.asarray(weighted_pdf.flatten())
  print("Shape of flattened scalarized pdf: ", pdf.shape)
  print("Running ErgCover on map scalarized with: ", w)
  print("Number of time steps: ", pbm.nA)
  controls, ergs, iters = ErgCover(pdf, n_agents, pbm.nA, pbm.s0, n_basis, pbm.pix, n_iter, ifDisplay, u_init, stop_eps, kkk, grad_criterion=True)
  print("Finished run scalarize method once")
  return controls, ergs, iters, weighted_pdf

def UniformGenerateWeights(n_weight, n_obj):
  """
  """
  out = list()
  if n_obj == 2:
    w1_list = np.linspace(0,1,n_weight)
    for w1 in w1_list:
      out.append( np.array( [w1,1-w1] ) )
  elif n_obj == 3:
    w_list = np.linspace(0,1,n_weight)
    for w1 in w_list:
      for w2 in w_list:
        if w1 + w2 > 1:
          break
        out.append( np.array( [1-w1-w2,w1,w2] ) )
  return out

def UniformGenerateWeights_Fourier(n_weight, n_obj):
  out = list()
  for i in range(n_weight):
    w_list = np.array(np.random.dirichlet(np.ones(n_obj),size=1))
    out.append(w_list)
  print("Returning fourier weight list of length: ", len(out))
  return out

def ErgodicDiff(calc1,calc2):
  """
  """
  lamk = calc1.lamk
  return np.sum(lamk*np.square(calc1.phik - calc2.phik))

def AdaptiveGenerateWeights(pbm, n_obj, delta=0.1):
  """
  """
  out = list()
  if n_obj == 2:
    diff12 = ErgodicDiff(pbm.calc1, pbm.calc2)
    print("[INFO] AdaptiveGenerateWeights, diff12 = ", diff12)
    dw12 = delta/diff12
    w1_list = np.linspace(0, 1, int(np.floor(1/dw12))+2) # at least two points
    for w1 in w1_list:
      out.append( np.array( [1-w1,w1] ) )
  elif n_obj == 3:
    diff12 = ErgodicDiff(pbm.calc1, pbm.calc2)
    dw12 = delta/diff12
    diff13 = ErgodicDiff(pbm.calc1, pbm.calc3)
    dw13 = delta/diff13
    print("[INFO] AdaptiveGenerateWeights, diff12 = ", diff12, " diff13 = ", diff13)
    w1_list = np.linspace(0, 1, int(np.floor(1/dw12))+2)
    flip = False
    for w1 in w1_list:
      w2_list = np.linspace(0, 1, int(np.floor(1/dw13))+2)
      if flip:
        w2_list = np.linspace(1, 0, int(np.floor(1/dw13))+2)
      for w2 in w2_list:
        if w1 + w2 > 1:
          flip = not flip
          break
        out.append( np.array( [1-w1-w2,w1,w2] ) )
  return out

def MOESSolveScalarize(pbm, n_agents, n_weight=11, n_basis=10, n_iter=1000, seqOptm=False, ifAdaptive=False, stop_eps=1e-5, ifDisplay=False, delta=0.1, comb=False):
  """
  n_weight defines the number of intervals to divide [0,1] for each weight component.
  """
  # print("In the beg of moes solve scalarize with number of info maps: ", len(pbm.pdfs))
  w1_list = np.linspace(0,1,n_weight)
  # n_obj=2
  n_obj = len(pbm.pdfs)
  if hasattr(pbm, 'pdf3'):
    n_obj=3
  if comb:
    weight_list = UniformGenerateWeights_Fourier(n_weight,len(pbm.pdfs))
  else:
    weight_list = UniformGenerateWeights(n_weight, n_obj)
  if ifAdaptive:
    weight_list = AdaptiveGenerateWeights(pbm, n_obj, delta)

  erg_metric_mat = np.zeros( (len(weight_list), n_obj) ) # Does this have to change with n_agents?
  u_list = list()
  pdf_list = list()
  erg_list = list()
  time_list = list()
  iter_list = list()
  u_init = None

  # print("Got weight list of length: ", len(weight_list))


  for j, weight in enumerate(weight_list):
    
    # print("weight = ", weight)
    if not seqOptm:
      u_init = None
    tnow = time.perf_counter()
    u, ergs, iters, weighted_pdf = RunScalarizeMethodOnce(pbm, n_agents, weight, n_basis, n_iter, ifDisplay, u_init, stop_eps, j, comb)
    
    used_time = time.perf_counter()-tnow
    time_list.append(used_time)
    u_list.append(u)
    erg_list.append(ergs)
    pdf_list.append(weighted_pdf)
    iter_list.append(iters)
    u_init = u_list[-1] # the solution from the last time.

    if comb:
      erg_vec = []
      for p in pbm.pdfs:
        p = jnp.asarray(p.flatten())
        EC = ergodic_metric.ErgCalc(p,1,pbm.nA,10,pbm.pix)
        e = EC.fourier_ergodic_loss(u,pbm.s0,pbm.nA)
        erg_vec.append(e)
    else:
      erg_vec = np.array([pbm.calc1.fourier_ergodic_loss(u, pbm.s0,pbm.nA), pbm.calc2.fourier_ergodic_loss(u, pbm.s0,pbm.nA)])
      if n_obj == 3:
        erg_vec = np.array([pbm.calc1.fourier_ergodic_loss(u, pbm.s0,pbm.nA), pbm.calc2.fourier_ergodic_loss(u, pbm.s0,pbm.nA), pbm.calc3.fourier_ergodic_loss(u, pbm.s0,pbm.nA)])
    erg_metric_mat[j,:] = erg_vec
  
  print("[INFO] MOESSolveScalarize use time = ", time_list)
  if comb:
    return erg_metric_mat, u_list, pdf_list, time_list, erg_list, iter_list, weight_list  
  return erg_metric_mat, u_list, pdf_list, time_list, erg_list, iter_list




