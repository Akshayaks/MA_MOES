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
from explicit_allocation import *

""""
First find an incumbent solution using greedy approach to get an "upperd_bound". The incumbent solution
will be the max(indv_ergodicities) for the greedy allocation combination

Maintain a second upper_bound "U2" using the EEE function values. (This would be a looser UB)

Initialize a "dominant" set to []. This will house the definitely bad assignments (not entire allocation)

The root node is [], i.e. no assignment has been done so far
Then each level corresponds to the assignment of each agent to the maps

For each node, if the the max(erg) so far is > upper bound then discard that node

**************************
within each node, can we calculate the ergodicity of the agent on the map combination we expect it to have the worst performance on?
This would be the second level of heuristic
***************************

Similarly, if the max(EEE) function value > U2 then too discard that node

***************************
To use EEE function as heuristic, we need to prove that it is a lower bound or under estimate of ergodicity
We also need to prove that it is consistent => How to draw the analogy to A* algorithm in this?
***************************

Continue this till the leaf nodes are reached. The number of levels in the tree will be equal to the number of agents.

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
		self.parent = None
		self.alive = True    #By default the node is alive (need to be explored)

	def __init__(self, agent, tasks, indv_erg, upper, U2, children, parent):
		self.agent = agent
		self.tasks = tasks
		self.indv_erg = indv_erg    #List of indv ergodicities on assigned maps
		self.upper = upper          #The maximum of the indv ergodicities so far
		self.U2 = U2                #The maximum of the indv EEE function values so far
		self.children = children    #List of the children of the Node
		self.parent = parent
		self.alive = True 

	def kill(self):
		self.alive = False    #Pruning/bounding

def greedy_alloc(problem, n_agents, n_scalar):
  #Allocate based on a circle around the agent and calculating the information in that region

  n_obj = len(problem.pdfs)

  sensor_footprint = 15
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
  print(x_range)
  print(y_range)

  #Calculate how much information agent 1 and agent 2 can cover when allocatted to map1 and map2 respectively and vice versa

  for i in range(n_agents):
  	xr = np.arange(x_range[i][0],x_range[i][1])
  	yr = np.arange(y_range[i][0],y_range[i][1])
  	print("agent: ", i)
  	print(xr,yr)

  	for p in range(n_obj):
  		print("objective: ", p)
  		for x in xr:
  			for y in yr:
  				print(x,y)
  				print(problem.pdfs[p][y][x])
  				agent_scores[i][p] += problem.pdfs[p][y][x]
  				print(agent_scores)
  		# pdb.set_trace()

  print("Agent scores: ", agent_scores)
  # pdb.set_trace()

  maps_assigned = np.zeros(n_obj)
  allocation = {}

  if n_agents >= n_obj:
  	for i in range(n_agents):
  		k = 0
  		erg = sorted(agent_scores[i][:], reverse=True)
  		found = False
  		while not found:
  			idx = agent_scores[i].tolist().index(erg[k])
  			if maps_assigned[idx]:
  				k += 1
  			else:
  				allocation[i] = [idx]
  				found = True
  				maps_assigned[idx] = 1 
  else:
  	print("No > Na")
  	agents_assigned = np.zeros(n_agents)
  	for j in range(n_agents):
  		allocation[j] = []
  	print("Allocation initialized")
  	for i in range(n_obj):
  		k = 0
  		erg = sorted([a[i] for a in agent_scores], reverse=True)
  		print("agent_scores: ",erg)
  		found = False
  		while not found:
  			idx = [a[i] for a in agent_scores].index(erg[k])
  			print("idx: ", idx)
  			if agents_assigned[idx] == 2:
  				k += 1
  			else:
  				allocation[idx] = allocation[idx] + [i]
  				found = True
  				agents_assigned[idx] += 1
  				print("allocation so far: ", allocation)
  print("The final allocations are as follows: ", allocation)
  pdb.set_trace()
  return allocation

def generate_alloc_nodes(root,curr_node,n_obj):
	maps_assigned = []
	while(curr_node):
		maps_assigned = maps_assigned + list(curr_node.tasks)
		curr_node = curr_node.parent
	maps_left = []
	for i in range(n_obj):
		if i not in maps_assigned:
			maps_left.append(i)
	assignments = []
	#arange should not be from 1. Eg: if there are only 2 agents then the second agent 
	for n in np.arange(1,len(maps_left)+1):
		assignments = assignments + list(itertools.combinations(maps_left, n))
	print("Assignments: ", assignments)
	# print(assignments)
	return assignments



def branch_and_bound(pbm_file, n_agents):

	start_time = time.time()
	pbm_file_complete = "./build_prob/test_cases/" + pbm_file
	
	problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)

	n_scalar = 3
	n_obj = len(problem.pdfs)
	print(generate_allocations(n_obj,n_agents))
	pdb.set_trace()

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

	display_map(problem,problem.s0,window=5)

	pdf_list = problem.pdfs

	print("Read start position as: ",problem.s0)

	print("Agent start positions allotted!")

	#Generate incumbent solution using Greedy approach
	incumbent = greedy_alloc(problem,n_agents,n_scalar)

	incumbent_erg = np.zeros(n_obj)

	final_allocation = {}

	nodes_pruned = 0

	#Find the upper bound for the incumbent solution
	for k,v in incumbent.items():
		if len(v) > 1:
			pdf = scalarize_minmax([pdf_list[a] for a in v],problem.s0[k*3:k*3+3],problem.nA)
		else:
			pdf = pdf_list[v[0]]

		pdf = jnp.asarray(pdf.flatten())
		
		#Just run ergodicity optimization for fixed iterations and see which agent achieves best ergodicity in that time
		control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)
		
		for p in v:
		  pdf_indv = jnp.asarray(pdf_list[p].flatten())
		  EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
		  incumbent_erg[p] = EC.fourier_ergodic_loss(control,problem.s0[3*k:3+3*k])

	upper = max(incumbent_erg)
	print("Incumbent allocation: ", incumbent)
	print("Incumber Ergodicities: ", incumbent_erg)
	print("Initial Upper: ", upper)
	pdb.set_trace()

	#Start the tree with the root node being [], blank assignment
	root = Node(None, [], [], np.inf, np.inf, [], None)
	print("Added root node")

	maps_assigned = np.zeros(n_obj)
	explore_node = [root]

	for i in range(n_agents):
		print("Looking at assignments for agent: ", i)
		pdb.set_trace()
		for curr_node in explore_node:
			alloc_comb = generate_alloc_nodes(root,curr_node,n_obj)
			print("Alloc_comb: ", alloc_comb)
			# pdb.set_trace()
			for a in alloc_comb:
				print("Looking at next assignment for node: ", i)
				print("curr_node: ", curr_node)
				print("Node created: ", Node(i, a, [], np.inf, np.inf, [], curr_node))
				# pdb.set_trace()
				node = Node(i, a, [], np.inf, np.inf, [], curr_node)
				print("Alloc assigned: ", a)
				# pdb.set_trace()
				prune = False
				if len(a) > 1:
					pdf = scalarize_minmax([pdf_list[j] for j in a],problem.s0[i*3:i*3+3],problem.nA)
				else:
					pdf = pdf_list[a[0]]

				pdf = jnp.asarray(pdf.flatten())

				#Just run ergodicity optimization for fixed iterations and see which agent achieves best ergodicity in that time
				control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*i:3+3*i], n_scalar, problem.pix, 1000, False, None, grad_criterion=False)

				for p in a:
					pdf_indv = jnp.asarray(pdf_list[p].flatten())
					EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
					erg = EC.fourier_ergodic_loss(control,problem.s0[3*i:3+3*i])
					print("erg: ", erg)
					# pdb.set_trace()
					if erg > upper:
						node.alive = False
						prune = True
						print("Don't explore further")
						nodes_pruned += 1 
						continue
					node.indv_erg.append(erg)
				print("Outside the assignments in a")
				# pdb.set_trace()
				if not prune:
					print("Not pruning this node")
					# pdb.set_trace()
					curr_node.children.append(node)
					print("node.indv_erg: ", node.indv_erg)
				# pdb.set_trace()

		new_explore_node = []
		for curr_node in explore_node:
			new_explore_node = new_explore_node + curr_node.children
		explore_node = new_explore_node

		if i == n_agents-1:
			print("Final agent assignments have been checked!")
			for e in explore_node:
				print(e.tasks)

	print("Number of nodes pruned: ", nodes_pruned)
	total_alloc = len(generate_allocations(n_obj,n_agents))
	print("Total number of nodes: ", total_alloc)
	print("Percentage of nodes pruned: ", nodes_pruned/total_alloc)
	return final_allocation

if __name__ == "__main__":
	pbm_file = "3_maps_example_3.pickle"
	n_agents = 2
	final_allocation = branch_and_bound(pbm_file,n_agents)
	print("Final allocation: ", final_allocation)


