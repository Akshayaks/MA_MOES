import matplotlib.pyplot as plt
import numpy as np

import common
import scalarize
from ergodic_coverage import ErgCover
import jax.numpy as jnp
import pdb
import copy
import ergodic_metric

import argparse
import time
import math
import os
from utils import *

""""
First find an incumbent solution using greedy approach to get an "upperd_bound". The incumbent solution
will be the max(indv_ergodicities) for the greedy allocation combination

Maintain a second upper_bound "U2" using the EEE function values.

Initialize a "dominant" set to []. This will house the definitely bad assignments (not entire allocation)

The root node is [], i.e. no assignment has been done so far
Then each level corresponds to the assignment of each agent to the maps

For each node, if the the max(erg) so far is > upper bound then discard that node 
Similarly, if the max(EEE) function value > U2 then too discard that node

Continue this till the leaf nodes are reached. The number of levels in the tree will be equal to the
number of agents.

**Need to prove that the EEE function is an admissible hueristic if we are using the EEE function to 
determine the bounding.
We can use EEE function for bounding and VES for branching?

"""

class Node:
	def __init__(self):
		self.agent = None
		self.tasks = []
		self.indv_erg = []   #List of indv ergodicities on assigned maps
		self.upper = None    #The maximum of the indv ergodicities so far
		self.U2 = None       #The maximum of the indv EEE function values so far
		self.children = []   #List of the children of the Node
		self.alive = True    #By default the node is alive (need to be explored)

	def __init__(self, agent, tasks, indv_erg, upper, U2, children):
		self.agent = agent
		self.tasks = tasks
		self.indv_erg = indv_erg    #List of indv ergodicities on assigned maps
		self.upper = upper          #The maximum of the indv ergodicities so far
		self.U2 = U2                #The maximum of the indv EEE function values so far
		self.children = children    #List of the children of the Node
		self.alive = True 

	def kill(self):
		self.alive = False    #Pruning/bounding

def greedy_alloc(problem, n_agents, n_scalar):
  #Allocate based on a circle around the agent and calculating the information in that region

  n_obj = len(problem.pdfs)

  sensor_footprint = 30
  agent_locs = []
  for i in range(n_agents):
    agent_locs.append((round(problem.s0[0+i*3]*100),round(problem.s0[1+i*3]*100)))
  print("Agent locations: ", agent_locs)

  x_range = []
  y_range = []

  for i in range(n_agents):
  	x_range.append((max(agent_locs[i][0]-sensor_footprint,0),min(agent_locs[i][0]+sensor_footprint,100)))
  	y_range.append((max(agent_locs[i][1]-sensor_footprint,0),min(agent_locs[i][1]+sensor_footprint,100)))

  agent_scores = np.zeros((n_agents,n_obj))

  #Calculate how much information agent 1 and agent 2 can cover when allocatted to map1 and map2 respectively and vice versa

  for i in range(n_agents):
  	xr = np.arange(x_range[i][0],x_range[i][1])
  	yr = np.arange(y_range[i][0],y_range[i][1])

  	for p in range(n_obj):
  		for x in xr:
  			for y in yr:
  				agent_scores[i][p] += problem.pdfs[p][x][y]

  print("Agent scores: ", agent_scores)

  maps_assigned = np.zeros(n_obj)
  allocation = {}

  for i in range(n_agents):
  	k = 0
  	erg = sorted(agent_scores[i][:], reverse=True)
  	found = False
  	while not found:
  		idx = agent_scores[i].tolist().index(erg[k])
  		if maps_assigned[idx]:
  			k += 1
  		else:
  			allocation[i] = idx 
  			found = True
  			maps_assigned[idx] = 1 
  print("The final allocations are as follows: ", allocation)

  return allocation

def branch_and_bound(pbm_file, n_agents):
	start_time = time.time()
	pbm_file_complete = "./build_prob/test_cases/" + pbm_file
	
	problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)

	n_scalar = 10
	n_obj = len(problem.pdfs)

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

	print("Read start position as: ",problem.s0)

	print("Agent start positions allotted!")

	#Generate incumbent solution using Greedy approach
	incumbent = greedy_alloc(problem,n_agents,n_scalar)

	#Find the upper bound for the incumbent solution


if __name__ == "__main__":
	pbm_file = "4_maps_example_3.pickle"
	n_agents = 4
	branch_and_bound(pbm_file,n_agents)


