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

np.random.seed(101)

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
		self.children = []   #List of the children of the Node
		self.parent = None
		self.alive = True    #By default the node is alive (need to be explored)

	def __init__(self, agent, tasks, indv_erg, upper, U2, children, parent):
		self.agent = agent
		self.tasks = tasks
		self.indv_erg = indv_erg    #List of indv ergodicities on assigned maps
		self.upper = upper          #The maximum of the indv ergodicities so far
		self.children = children    #List of the children of the Node
		self.parent = parent
		self.alive = True 

	def kill(self):
		self.alive = False    #Pruning/bounding

def greedy_alloc(problem, n_agents, n_scalar, node = None):
  #Allocate based on a circle around the agent and calculating the information in that region

  n_obj = len(problem.pdfs)
  agents_allotted = []
  maps_allotted = []
  if node:
  	current_node = node
  	while current_node.parent:
  		agents_allotted.append(current_node.agent)
  		maps_allotted = maps_allotted + list(current_node.tasks)
  		current_node = current_node.parent

  print("\nAgents already allotted: ", agents_allotted)
  print("\nMaps already assigned: ", maps_allotted)
  if len(agents_allotted) == n_agents:
  	print("all agents are allotted")
  	return -1
  #pd.set_trace()

  sensor_footprint = 15
  agent_locs = []
  for i in range(n_agents):
    if i in agents_allotted:
      agent_locs.append((-1,-1))
    else:
      agent_locs.append((round(problem.s0[0+i*3]*100),round(problem.s0[1+i*3]*100)))
  print("Agent locations: ", agent_locs)

  x_range = []
  y_range = []

  for i in range(n_agents):
  	if i in agents_allotted:
  		x_range.append((-1,-1))
  		y_range.append((-1,-1))
  	else:
  		x_range.append((max(agent_locs[i][0]-sensor_footprint,0),min(agent_locs[i][0]+sensor_footprint,100)))
  		y_range.append((max(agent_locs[i][1]-sensor_footprint,0),min(agent_locs[i][1]+sensor_footprint,100)))

  agent_scores = np.zeros((n_agents,n_obj))
  print(x_range)
  print(y_range)

  #Calculate how much information agent 1 and agent 2 can cover when allocatted to map1 and map2 respectively and vice versa

  for i in range(n_agents):
  	if i in agents_allotted:
  		continue
  	xr = np.arange(x_range[i][0],x_range[i][1])
  	yr = np.arange(y_range[i][0],y_range[i][1])
  	print("agent: ", i)
  	print(xr,yr)

  	for p in range(n_obj):
  		if p in maps_allotted:
  			continue
  		print("objective: ", p)
  		for x in xr:
  			for y in yr:
  				print(x,y)
  				print(problem.pdfs[p][y][x])
  				agent_scores[i][p] += problem.pdfs[p][y][x]
  				print(agent_scores)
  		# #pd.set_trace()

  print("Agent scores: ", agent_scores)
  #pd.set_trace()

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
  		if i in maps_allotted:
  			continue
  		k = 0
  		erg = sorted([a[i] for a in agent_scores], reverse=True)
  		print("agent_scores: ",erg)
  		found = False
  		while not found:
  			idx = [a[i] for a in agent_scores].index(erg[k])
  			print("idx: ", idx)
  			if agents_assigned[idx] == 3 or idx in agents_allotted:
  				k += 1
  			else:
  				allocation[idx] = allocation[idx] + [i]
  				found = True
  				agents_assigned[idx] += 1
  				print("allocation so far: ", allocation)
  print("The final allocations are as follows: ", allocation)
  #pd.set_trace()
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

def find_best_allocation(root,values,alloc,indv_erg):
    # list to store path
    if root == None:
    	return 
    if len(root.children) == 0:
    	print("Reached leaf node")
    	path = {}
    	max_erg = []
    	values.append(root)
    	path[values[0].agent] = values[0].tasks
    	max_erg += values[0].indv_erg
    	print("Path before adding all elements in values: ", path)
    	# print("values: ", values)
    	for i in np.arange(1,len(values)):
    		path[values[i].agent] = values[i].tasks
    		max_erg += values[i].indv_erg
    	alloc.append(path)
    	indv_erg.append(max_erg)
    	print("Path found: ", path)
    	values.pop()
    	return
    print(str(root.agent)+" has "+str(len(root.children))+" children!")
    values.append(root)
    # print("Values: ", values)
    for child in root.children:
    	find_best_allocation(child,values,alloc,indv_erg)
    values.pop()

def branch_and_bound(pbm_file, n_agents):

	start_time = time.time()
	pbm_file_complete = "./build_prob/random_maps/" + pbm_file
	
	problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)

	n_scalar = 10
	n_obj = len(problem.pdfs)
	if n_obj > 6:
		print("Too many objectives: ", n_obj)
		return [],0

	problem.nA = 100 
	nA = problem.nA

	start_pos = np.load("start_pos_random_2_agents.npy",allow_pickle=True)

	problem.s0 = start_pos.item().get(pbm_file)
	print("Read start position as: ",problem.s0)

	print("Agent start positions allotted!")

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
		pdf = np.zeros((100,100))
		if len(v) > 1:
			for a in v:
			  pdf += (1/len(v))*pdf_list[a]
			# pdf = scalarize_minmax([pdf_list[a] for a in v],problem.s0[k*3:k*3+3],problem.nA)
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
	#pd.set_trace()

	#Start the tree with the root node being [], blank assignment
	root = Node(None, [], [], np.inf, np.inf, [], None)
	print("Added root node")

	maps_assigned = np.zeros(n_obj)
	explore_node = [root]

	for i in range(n_agents):
		print("Looking at assignments for agent: ", i)
		#pd.set_trace()
		new_explore_node = []
		for curr_node in explore_node:
			alloc_comb = generate_alloc_nodes(root,curr_node,n_obj)
			print("Alloc_comb: ", alloc_comb)
			for a in alloc_comb:
				print("Looking at next assignment for node: ", i)
				print("curr_node: ", curr_node)
				node = Node(i, a, [], np.inf, np.inf, [], curr_node)
				print("Alloc assigned: ", a)
				# #pd.set_trace()
				prune = False
				if len(a) > 1:
					pdf = np.zeros((100,100))
					for j in a:
					  pdf += (1/len(a))*pdf_list[j]
					# pdf = scalarize_minmax([pdf_list[j] for j in a],problem.s0[i*3:i*3+3],problem.nA)
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
					# #pd.set_trace()
					if erg > upper:
						node.alive = False
						prune = True
						print("Don't explore further")
						nodes_pruned += 1 
						continue
					node.indv_erg.append(erg)
			
				if not prune:
					print("Not pruning this node")
					# #pd.set_trace()
					curr_node.children.append(node)
					print("node.indv_erg: ", node.indv_erg)
					new_explore_node.append(node)

		explore_node = new_explore_node

		if i == n_agents-1:
			print("Final agent assignments have been checked!")
			for e in explore_node:
				print(e.tasks)
			# pdb.set_trace()

	print("Number of nodes pruned: ", nodes_pruned)
	total_alloc = len(generate_allocations(n_obj,n_agents))
	print("Total number of nodes: ", total_alloc)
	print("Percentage of nodes pruned: ", nodes_pruned/total_alloc)
	runtime = time.time() - start_time
	return root, problem.s0, runtime, nodes_pruned/total_alloc

if __name__ == "__main__":
	# pbm_file = "3_maps_example_3.pickle"
	# n_agents = 2
	# final_allocation = branch_and_bound(pbm_file,n_agents)
	# print("Final allocation: ", final_allocation)

	parser = argparse.ArgumentParser()
	# parser.add_argument('--method', type=str, required=True, help="Method to run")
	parser.add_argument('--test_folder', type=str, required=True, help="Folder with test cases", default="./build/test_cases/")

	args = parser.parse_args()
	folder = args.test_folder
	n_agents = 2

	run_times = {}
	best_allocs = {}
	nodes_pruned = {}
	# start_positions = gen_start_pos(folder,2)
	for pbm_file in os.listdir(folder):
		root,start_pos,run_time,per_nodes_pruned = branch_and_bound(pbm_file,2)

		values = []
		alloc = []
		indv_erg = []
		find_best_allocation(root,values,alloc,indv_erg)
		print("All paths found: ", alloc)
		print("Individual ergodicities: ", indv_erg)

		#Among all the allocations found, pick the one with min max erg
		max_erg = []
		for e in indv_erg:
			print("e: ", e)
			print("len(e): ", len(e))
			if len(e) < n_agents:
				max_erg.append(100)
			else:
				max_erg.append(max(e))

		print("Max ergodicities: ", max_erg)
		min_idx = np.argmin(max_erg)

		best_alloc = alloc[min_idx]

		print("The best allocation according to minmax metric: ", best_alloc)
		print("Best allocation: ", best_alloc)
		print("Runtime: ", run_time)
		# pdb.set_trace()

		run_times[pbm_file] = run_time
		best_allocs[pbm_file] = best_alloc
		nodes_pruned[pbm_file] = per_nodes_pruned
		np.save("BB_random_maps_runtime_2_agents.npy", run_times)
		np.save("Best_alloc_BB_random_maps_2_agents.npy",best_allocs)
		np.save("nodes_pruned_BB_random_maps_2_agents.npy",nodes_pruned)


