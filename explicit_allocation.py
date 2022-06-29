import matplotlib.pyplot as plt
import numpy as np

import common
import scalarize
from ergodic_coverage import ErgCover
import jax.numpy as jnp
import pdb
import copy
import ergodic_metric

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.stats import wasserstein_distance

import argparse
import time
import math
import os
from utils import *

'''
Generate the pareto front using MOES and pick the best weight to scalarize maps
'''
def scalarize_minmax(pdf_list, s0, nA):
  # print("In scalarize minimax!")
  dic = {"s0": s0, "nA": nA, "pdfs": pdf_list, "nPixel": 100}
  n_agents = 1
  pix = 100
  n_scalar = 10

  pbm = common.MOESProblem_Fourier()
  pbm.initFromDic(dic,n_agents)

  erg_mat_scala, u_list_scala, pdf_list_scala, time_scala, erg_list_scala, iter_list_scala, weight_list = \
  scalarize.MOESSolveScalarize(pbm, n_agents, n_scalar, 10, 1000, seqOptm=False, stop_eps=3e-3,comb=True)
  
  erg_max = np.amax(erg_mat_scala,axis=1)
  idx_min = np.argmin(erg_max)

  i = 0
  pdf_scalarized = np.zeros((100,100))
  for p in pdf_list:
    pdf_scalarized += p*weight_list[idx_min][0][i]
    i += 1 
  return pdf_scalarized


#Evaluate all combinations of allocation
def main_run_comb_allocation(pbm_file,n_agents):

  start_time = time.time()
  pbm_file_complete = "./build_prob/test_cases/" + pbm_file
  
  problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)

  n_scalar = 10
  n_obj = len(problem.pdfs)

  problem.nA = 100 
  nA = problem.nA
  
  start_pos = np.load("start_pos.npy",allow_pickle=True)

  problem.s0 = start_pos.item().get(pbm_file)
  print("Read start position as: ",problem.s0)

  print("Agent start positions allotted!")

  alloc_comb = generate_allocations(n_obj,n_agents)

  erg_mat = np.zeros((len(alloc_comb),n_obj))   #For each allocation, we want the individual ergodicities

  pdf_list = problem.pdfs
  trajectory = []

  for idx,alloc in enumerate(alloc_comb):
    print(alloc)
    for i in range(n_agents):
      pdf = np.zeros((100,100))
      if len(alloc[i]) > 1:
        # for a in alloc[i]:
          # pdf += (1/len(alloc[i]))*pdf_list[a]
        pdf = scalarize_minmax([pdf_list[a] for a in alloc[i]],problem.s0[i*3:i*3+3],problem.nA)
      else:
        pdf = pdf_list[alloc[i][0]]
      
      pdf = jnp.asarray(pdf.flatten())
      print(problem.s0)
      
      #Just run ergodicity optimization for fixed iterations and see which agent achieves best ergodicity in that time
      control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*i:3+3*i], n_scalar, problem.pix, 1000, False, None, grad_criterion=False)
      
      for p in alloc[i]:
        pdf_indv = jnp.asarray(pdf_list[p].flatten())
        EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
        erg_mat[idx][p] = EC.fourier_ergodic_loss(control,problem.s0[3*i:3+3*i])

  runtime = time.time() - start_time
  
  max_array = []
  for i in range(len(alloc_comb)):
    max_array.append(max(erg_mat[i][:]))
  best_alloc = np.argmin(max_array)

  display_map(problem,problem.s0,pbm_file,title="Best Allocation: "+str(alloc_comb[best_alloc]))
  
  return best_alloc,runtime


def get_win_sizes(problem,n_agents,n_scalar):
  n = len(problem.pdfs)
  window = np.zeros((n,n_agents))
  for i in range(n):
    for j in range(n_agents):
      pdf = problem.pdfs[i]
      EC_whole = ergodic_metric.ErgCalc(pdf.flatten(),1,problem.nA,n_scalar,problem.pix)
      w_size = 0
      for w_size in np.arange(5,90,5):
        ###   To create a window around the agent ###
        x,y = problem.s0[j*3:j*3+2]*100
        x = round(x)
        y = round(y)
        print("w_size: ", w_size)
        print("x,y: ", x,y)
        h1 = max(0,y-w_size)
        h2 = min(100,y+w_size)
        w1 = max(0,x-w_size)
        w2 = min(100,x+w_size)
        print("h1,h2,w1,w2: ", h1,h2,w1,w2)
        
        pdf_zeroed = np.zeros_like(pdf)
        pdf_zeroed[h1:h2,w1:w2] = pdf[h1:h2,w1:w2]
        EC_window = ergodic_metric.ErgCalc(pdf_zeroed.flatten(),1,problem.nA,n_scalar,problem.pix)
        FC_diff = np.linalg.norm(EC_whole.phik - EC_window.phik)
        print("fc_diff: ",FC_diff)
        if FC_diff < 1:
          break
      window[i][j] = w_size
  print("Windows calculated: ", window)
  w_max = window.max()
  return window, w_max



def main_run_win_alloc(pbm_file,n_agents,crop=False):
  
  pbm_file_complete = "./build/test_cases/"+pbm_file
  
  problem = common.LoadProblem(pbm_file_complete, n_agents)

  n_scalar = 10

  problem.nA = 100 
  nA = problem.nA
  n_obj = len(problem.pdfs)

  start_pos = np.load("start_pos.npy",allow_pickle=True)

  problem.s0 = start_pos.item().get(pbm_file)

  window, w_max = get_win_sizes(problem,n_agents,n_scalar)
  start_time = time.time() 

  print("Agent start positions allotted!")

  alloc_comb = generate_allocations(n_obj,n_agents)

  pdf_list = problem.pdfs

  erg_mat = np.zeros((len(alloc_comb),n_obj))   #For each allocation, we want the individual ergodicities

  trajectory = []

  for idx,alloc in enumerate(alloc_comb):
    print(alloc)
    
    for i in range(n_agents):
      ###   To create a window around the agent ###
      x,y = problem.s0[i*3:i*3+2]*100
      x = round(x)
      y = round(y)

      pdf = np.zeros_like(pdf_list[0])

      if len(alloc[i]) > 1:
        print("Number of maps allotted: ", len(alloc[i]))
        pdf = scalarize_minmax([pdf_list[a] for a in alloc[i]],problem.s0[i*3:i*3+3],problem.nA)
        # for a in alloc[i]:
        #   pdf += (1/len(alloc[i]))*pdf_list[a]
        w_size = int(w_max)
        h1 = max(0,y-w_size)
        h2 = min(100,y+w_size)
        w1 = max(0,x-w_size)
        w2 = min(100,x+w_size)
        
      else:
        print("Only one map allotted")
        pdf = pdf_list[alloc[i][0]]
        w_size = int(window[alloc[i][0]][i])
        h1 = max(0,y-w_size)
        h2 = min(100,y+w_size)
        w1 = max(0,x-w_size)
        w2 = min(100,x+w_size)

      
      start_pos0 = np.array([x-w1,y-h1,0])/(h2-h1)
      
      if crop:
        pdf_zeroed = pdf[h1:h2,w1:w2]
      else:
        pdf_zeroed = np.zeros_like(pdf_list[0])
        pdf_zeroed[h1:h2,w1:w2] = pdf[h1:h2,w1:w2]
      
      pdf_flat = jnp.asarray(pdf_zeroed.flatten())
      
      #Just run ergodicity optimization for information map of original size or the cropped map
      # control, erg, _ = ErgCover(pdf_flat, 1, problem.nA, problem.s0[i*3:i*3+3], n_scalar, problem.pix, 1000, False, None, grad_criterion=False)
      control, erg, _ = ErgCover(pdf_flat, 1, problem.nA, start_pos0, n_scalar, [w2-w1,h2-h1], 1000, False, None, grad_criterion=False)
      print("Ergodicity inside the window: ", erg[-1])

      for p in alloc[i]:
        pdf_indv = jnp.asarray(pdf_list[p].flatten())
        EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
        erg_mat[idx][p] = EC.fourier_ergodic_loss(control,problem.s0[3*i:3+3*i])
        print("Ergodicity on the entire map: ", erg_mat[idx][p])

  runtime = time.time() - start_time
  print("Execution time of window version of exhaustive algorithm: ", runtime)
  
  max_array = []
  for i in range(len(alloc_comb)):
    max_array.append(max(erg_mat[i][:]))

  best_alloc = np.argmin(max_array)
  display_map(problem,problem.s0,pbm_file,title="Best Allocation: "+str(alloc_comb[best_alloc]))
  
  return best_alloc,runtime

def detect_peaks(image):
      """
      Takes an image and detect the peaks usingthe local maximum filter.
      Returns a boolean mask of the peaks (i.e. 1 when
      the pixel's value is the neighborhood maximum, 0 otherwise)
      """

      # define an 8-connected neighborhood
      neighborhood = generate_binary_structure(2,2)

      #apply the local maximum filter; all pixel of maximal value 
      #in their neighborhood are set to 1
      local_max = maximum_filter(image, footprint=neighborhood)==image

      plt.imshow(local_max,origin='lower')
      plt.show()
      #local_max is a mask that contains the peaks we are 
      #looking for, but also the background.
      #In order to isolate the peaks we must remove the background from the mask.

      #we create the mask of the background
      background = (image==0)

      #a little technicality: we must erode the background in order to 
      #successfully subtract it form local_max, otherwise a line will 
      #appear along the background border (artifact of the local maximum filter)
      eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

      #we obtain the final mask, containing only peaks, 
      #by removing the background from the local_max mask (xor operation)
      detected_peaks = local_max ^ eroded_background

      return detected_peaks

def case_3_allocation():
  n_agents = 5
  n_obj = 2 

  pbm_file = "build/instances/MOES-O2-peaks_pix_100_multimodal.pickle"
  
  problem = common.LoadProblem(pbm_file, n_agents)

  n_scalar = 10

  problem.nA = 100 
  nA = problem.nA
  
  #Generate random starting positions for the agents
  pos = np.random.uniform(0,1,2*n_agents)

  problem.s0 = []
  k = 0
  for i in range(n_agents):
    problem.s0.append(pos[k])
    problem.s0.append(pos[k+1])
    problem.s0.append(0)
    k += 2

  problem.s0 = np.array(problem.s0)

  print("Agent start positions allotted!")

  ###  Display the three information maps  ##############

  x = np.linspace(0,100,num=100)
  y = np.linspace(0,100,num=100)
  X,Y = np.meshgrid(x,y)

  fig, axs = plt.subplots(2, 3)
  axs[0,0].contourf(X, Y, problem.pdf1, levels=np.linspace(np.min(problem.pdf1), np.max(problem.pdf1),100), cmap='gray')
  axs[0,0].set_title('First Info Map')
  axs[0,1].contourf(X, Y, problem.pdf2, levels=np.linspace(np.min(problem.pdf2), np.max(problem.pdf2),100), cmap='gray')
  axs[0,1].set_title('Second Info Map')
  # axs[0,2].contourf(X, Y, problem.pdf3, levels=np.linspace(np.min(problem.pdf3), np.max(problem.pdf3),100), cmap='gray')
  # axs[0,2].set_title('Third Info Map')

  axs[0,0].plot(problem.s0[0]*100,problem.s0[1]*100,"go--")
  axs[0,0].plot(problem.s0[3]*100,problem.s0[4]*100,"ro--")
  axs[0,0].plot(problem.s0[6]*100,problem.s0[7]*100,"bo--")
  axs[0,0].plot(problem.s0[9]*100,problem.s0[10]*100,"wo--")
  axs[0,0].plot(problem.s0[12]*100,problem.s0[13]*100,"yo--")

  axs[0,1].plot(problem.s0[0]*100,problem.s0[1]*100,"go--")
  axs[0,1].plot(problem.s0[3]*100,problem.s0[4]*100,"ro--")
  axs[0,1].plot(problem.s0[6]*100,problem.s0[7]*100,"bo--")
  axs[0,1].plot(problem.s0[9]*100,problem.s0[10]*100,"wo--")
  axs[0,1].plot(problem.s0[12]*100,problem.s0[13]*100,"yo--")

  # axs[0,2].plot(problem.s0[0]*100,problem.s0[1]*100,"go--")
  # axs[0,2].plot(problem.s0[3]*100,problem.s0[4]*100,"ro--")
  # axs[0,2].plot(problem.s0[6]*100,problem.s0[7]*100,"bo--")
  # axs[0,2].plot(problem.s0[9]*100,problem.s0[10]*100,"wo--")
  # axs[0,2].plot(problem.s0[12]*100,problem.s0[13]*100,"yo--")

  plt.pause(1)
  plt.show()
  

  pdf_list = [problem.pdf1,problem.pdf2] #,problem.pdf3]

  peaks_array = []
  for pdf in pdf_list:
    p = []
    peaks = detect_peaks(pdf)
    print("peaks shape: ", peaks.shape)
    idx = np.argwhere(peaks == True)
    print("idx: ", idx)
    p.append(pdf[idx])
    peaks_array.append(p)
  print("Peaks array: ", peaks_array)

  uniform_dist = (np.ones((100,100))*0.0001).flatten()

  pdf1 = pdf_list[0].flatten()
  pdf2 = pdf_list[1].flatten()
  dist = wasserstein_distance(pdf1,pdf2)
  print("wasserstein_distance between the two maps: ", dist)
  print("wasserstein_distance between map 1 and uniform dist: ", wasserstein_distance(pdf1,uniform_dist))
  print("wasserstein_distance between map 2 and uniform dist: ", wasserstein_distance(pdf2,uniform_dist))

  return


def greedy_alloc():
  #Allocate based on a circle around the agent and calculating the information in that region
  n_agents = 2
  n_obj = 2 

  pbm_file = "build/instances/MOES-O2-nA_100_nGau_1_pix_100_simple.pickle"
  
  problem = common.LoadProblem(pbm_file, n_agents)

  n_scalar = 10

  problem.nA = 200 
  nA = problem.nA
  
  #Generate random starting positions for the agents
  pos = np.random.uniform(0,1,2*n_agents)

  problem.s0 = []
  k = 0
  for i in range(n_agents):
    problem.s0.append(pos[k])
    problem.s0.append(pos[k+1])
    problem.s0.append(0)
    k += 2

  problem.s0 = np.array(problem.s0)

  print("Agent start positions allotted!")

  sensor_footprint = 30
  agent_locs = []
  for i in range(n_agents):
    agent_locs.append((round(problem.s0[0+i*3]*100),round(problem.s0[1+i*3]*100)))
  print("Agent locations: ", agent_locs)

  #Looking at the area defined by a circle of radius sensor_footprint for each agent
  x1 = np.arange(max(agent_locs[0][0]-sensor_footprint,0),min(agent_locs[0][0]+sensor_footprint,100))
  y1 = np.arange(max(agent_locs[0][1]-sensor_footprint,0),min(agent_locs[0][1]+sensor_footprint,100))

  x2 = np.arange(max(agent_locs[1][0]-sensor_footprint,0),min(agent_locs[1][0]+sensor_footprint,100))
  y2 = np.arange(max(agent_locs[1][1]-sensor_footprint,0),min(agent_locs[1][1]+sensor_footprint,100))

  score_agent1 = [0,0]
  score_agent2 = [0,0]

  #Calculate how much information agent 1 and agent 2 can cover when allocatted to map1 and map2 respectively and vice versa

  for x in x1:
    for y in y1:
      score_agent1[0] += problem.pdf1[x][y]
      score_agent1[1] += problem.pdf2[x][y]

  print("Agent 1 score: ", score_agent1)

  for x in x2:
    for y in y2:
      score_agent2[0] += problem.pdf1[x][y]
      score_agent2[1] += problem.pdf2[x][y]

  print("Agent 2 score: ", score_agent2)

  agent1_allocation = np.argmax(score_agent1)
  agent2_allocation = np.argmax(score_agent2)

  #force random allocation incase both agents get allotted to the same map
  if agent1_allocation == agent2_allocation: 
    agent1_allocation = 0 
    agent2_allocation = 1

  p1 = jnp.asarray(problem.pdf1.flatten())
  p2 = jnp.asarray(problem.pdf2.flatten())

  #Vanilla ergodic coverage depending on the allocation
  if agent1_allocation == 0:
    control1, erg1, iter1 = ErgCover(p1, 1, problem.nA, problem.s0[0:3], n_scalar, problem.pix, 1000, False, None, 1e-3, 0)
    control2, erg2, iter2 = ErgCover(p2, 1, problem.nA, problem.s0[3:6], n_scalar, problem.pix, 1000, False, None, 1e-3, 0)
    EC1 = ergodic_metric.ErgCalc(p1,1,problem.nA,n_scalar,problem.pix)
    EC2 = ergodic_metric.ErgCalc(p2,1,problem.nA,n_scalar,problem.pix)
    e1 = EC1.fourier_ergodic_loss(control1, problem.s0[0:3], True)
    e2 = EC2.fourier_ergodic_loss(control2, problem.s0[3:6], True)
  else:
    control1, erg1, iter1 = ErgCover(p1, 1, problem.nA, problem.s0[3:6], n_scalar, problem.pix, 1000, False, None, 1e-3, 0)
    control2, erg2, iter2 = ErgCover(p2, 1, problem.nA, problem.s0[0:3], n_scalar, problem.pix, 1000, False, None, 1e-3, 0)
    EC1 = ergodic_metric.ErgCalc(p1,1,problem.nA,n_scalar,problem.pix)
    EC2 = ergodic_metric.ErgCalc(p2,1,problem.nA,n_scalar,problem.pix)
    e1 = EC1.fourier_ergodic_loss(control1, problem.s0[3:6], True)
    e2 = EC2.fourier_ergodic_loss(control2, problem.s0[0:3], True)
  

  print("Ergodicity for map1: ", e1)
  print("Ergodicity for map2: ", e2)

  x = np.linspace(0,100,num=100)
  y = np.linspace(0,100,num=100)
  X,Y = np.meshgrid(x,y)

  fig, axs = plt.subplots(2, 2)
  axs[0, 0].contourf(X, Y, problem.pdf1, levels=np.linspace(np.min(problem.pdf1), np.max(problem.pdf1),100), cmap='gray')
  axs[0, 0].set_title('First Info Map')
  axs[0, 1].contourf(X, Y, problem.pdf2, levels=np.linspace(np.min(problem.pdf2), np.max(problem.pdf2),100), cmap='gray')
  axs[0, 1].set_title('Second Info Map')
  

  _, tj1 = ergodic_metric.GetTrajXY(control1, problem.s0[:3])
  _, tj2 = ergodic_metric.GetTrajXY(control2, problem.s0[3:6])


  problem.s0 = problem.s0*100
  
  #Plot the starting points
  if agent1_allocation == 0:
    axs[0, 0].plot(problem.s0[0],problem.s0[1],"wo--")
    axs[0, 1].plot(problem.s0[3],problem.s0[4],"wo--")
  else:
    axs[0, 1].plot(problem.s0[0],problem.s0[1],"wo--")
    axs[0, 0].plot(problem.s0[3],problem.s0[4],"wo--")


  #Plot the trajectories
  if agent1_allocation == 0:
    axs[0, 0].plot(tj1[:,0]*100,tj1[:,1]*100)
    axs[0, 1].plot(tj2[:,0]*100,tj2[:,1]*100)
  else:
    axs[0, 1].plot(tj1[:,0]*100,tj1[:,1]*100)
    axs[0, 0].plot(tj2[:,0]*100,tj2[:,1]*100)
 
  plt.show()
  return

def H_function(pdf,s0,h=0):

  #Returns the EEE function which is a scalar value for a given map and start location

  H,W = pdf.shape
  print("Pdf shape: ", H,W)
  print(s0[0]*100,s0[1]*100)
  print(s0[0],s0[1])

  x = np.linspace(0,100,num=100)
  y = np.linspace(0,100,num=100)
  X,Y = np.meshgrid(x,y)

  wt_dist = np.zeros_like(pdf)
  for i in range(pdf.shape[0]):
    for j in range(pdf.shape[1]):
      dist = np.sqrt((s0[0]*100 - i)**2 +  (s0[1]*100 - j)**2)
      if dist != 0:
        wt_dist[i][j] = pdf[i][j]/dist
  h_value = sum(sum(wt_dist))
  return h_value


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--method', type=str, required=True, help="Method to run")

  args = parser.parse_args()
  pbm_file = "2_maps_example_4.pickle"
  
  if args.method == "case3":
    case_3_allocation()
  elif args.method == "MOES":
    alloc,runtime = main_run_comb_allocation(pbm_file,2)
  elif args.method == "EEE":
    allocation_h_func()
  elif args.method == "greedy":
    greedy_alloc()
  else:
    print("Enter a valid method!")

