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
import copy

np.random.seed(100)

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
		self.depth = 0

	def __init__(self, agent, tasks, indv_erg, upper, U2, children, parent):
		self.agent = agent
		self.tasks = tasks
		self.indv_erg = indv_erg    #List of indv ergodicities on assigned maps
		self.upper = upper          #The maximum of the indv ergodicities so far
		self.children = children    #List of the children of the Node
		self.parent = parent
		self.alive = True
		if parent:
			self.depth = parent.depth+1
		else:
			self.depth = 0

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

  print("Number of agents: ", n_agents)
  print("Number of objectives: ", n_obj)
  # pdb.set_trace()
  agent_scores = np.zeros((n_agents,n_obj))
  # print(x_range)
  # print(y_range)

  #Calculate how much information agent 1 and agent 2 can cover when allocatted to map1 and map2 respectively and vice versa

  for i in range(n_agents):
  	if i in agents_allotted:
  		continue
  	xr = np.arange(x_range[i][0],x_range[i][1])
  	yr = np.arange(y_range[i][0],y_range[i][1])
  	# print("agent: ", i)
  	# print(xr,yr)

  	for p in range(n_obj):
  		if p in maps_allotted:
  			continue
  		# print("objective: ", p)
  		for x in xr:
  			for y in yr:
  				# print(x,y)
  				# print(problem.pdfs[p][y][x])
  				agent_scores[i][p] += problem.pdfs[p][y][x]
  				# print(agent_scores)
  		# #pd.set_trace()

  # print("Agent scores: ", agent_scores)
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
  	# print("No > Na")
  	agents_assigned = np.zeros(n_agents)
  	for j in range(n_agents):
  		allocation[j] = []
  	# print("Allocation initialized")
  	for i in range(n_obj):
  		if i in maps_allotted:
  			continue
  		k = 0
  		erg = sorted([a[i] for a in agent_scores], reverse=True)
  		print("agent_scores: ",erg)
  		found = False
  		while not found:
  			idx = [a[i] for a in agent_scores].index(erg[k])
  			# print("idx: ", idx)
  			#Find the bug in this line
  			# print("agents_allotted: ", agents_allotted)
  			# print("agent assigned[idx]: ", agents_assigned[idx])
  			# print("n_obj-i-n_agents: ", n_obj-i-n_agents)
  			zero_map_agents = list(agents_assigned).count(0)
  			if (agents_assigned[idx] > 0 and n_obj-i == zero_map_agents) or idx in agents_allotted:
  				k += 1
  			else:
  				allocation[idx] = allocation[idx] + [i]
  				found = True
  				agents_assigned[idx] += 1
  				print("allocation so far: ", allocation)
  print("The final allocations are as follows: ", allocation)
  #pd.set_trace()
  return allocation

def generate_alloc_nodes(root,curr_node,n_obj,num_agents):
	maps_assigned = []
	temp = curr_node

	while(curr_node):
		maps_assigned = maps_assigned + list(curr_node.tasks)
		curr_node = curr_node.parent
	maps_left = []
	for i in range(n_obj):
		if i not in maps_assigned:
			maps_left.append(i)
	assignments = []
	#arange should not be from 1. Eg: if there are only 2 agents then the second agent

	print("Current node depth: ", temp.depth)

	if temp.depth+1 == num_agents:
		start = len(maps_left)
	else:
		start = 1
	# print("Maps left: ", maps_left)
	# print("start: ", start)
	# print("till: ", len(maps_left)-(num_agents-temp.depth-1))

	for n in np.arange(start,len(maps_left)-(num_agents-temp.depth-1)+1):
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

 
# Helper function to print path from root
# to leaf in binary tree
def printPathsRec(root, path, pathLen):
     
    # Base condition - if binary tree is
    # empty return
    print("Inside printPathRec")
    if len(root.children) == 0:
        return
    print("Root tasks not empty")
    for c in root.children:
    	print(c.tasks)
    	print("\n")
 
    # add current root's data into
    # path_ar list
     
    # if length of list is gre
    if(len(path) > pathLen):
        path[pathLen] = root.tasks
    else:
        path.append(root.tasks)
    print("path: ", path)
 
    # increment pathLen by 1
    pathLen = pathLen + 1
 
    if len(root.children) == 0:
         
        # leaf node then print the list
        printArray(path, pathLen)
    else:
        # try for left and right subtree
        # print("Recursive calls to root's children!")
        for c in root.children:
        	printPathsRec(c, path, pathLen)
 
# Helper function to print list in which
# root-to-leaf path is stored
def printArray(ints, len):
	print("Inside printArray!!!!")
	for i in ints[0 : len]:
		print(i," ",end="")
	print()

def branch_and_bound(pbm_file, n_agents, start=[-1]):

	start_time = time.time()
	pbm_file_complete = "/home/akshaya/MA_MOES/build_prob/instances/" + pbm_file
	
	problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)

	n_scalar = 10
	n_obj = len(problem.pdfs)
	print("number of obj: ", n_obj)

	problem.nA = 100 
	nA = problem.nA

	#Generate random starting positions for the agents
	pos = np.random.uniform(0,1,2*n_agents)

	if start[0] != -1:
		problem.s0 = start
	else:
		problem.s0 = []
		k = 0
		for i in range(n_agents):
		  problem.s0.append(pos[k])
		  problem.s0.append(pos[k+1])
		  problem.s0.append(0)
		  k += 2

		problem.s0 = np.array(problem.s0)

	# display_map(problem,problem.s0,window=5)

	pdf_list = problem.pdfs

	print("Read start position as: ",problem.s0)

	#Generate incumbent solution using Greedy approach
	incumbent = greedy_alloc(problem,n_agents,n_scalar)

	incumbent_erg = np.zeros(n_obj)

	final_allocation = {}
	nodes_pruned = 0

	#Find the upper bound for the incumbent solution
	for k,v in incumbent.items():
		print("v: ", v)
		if len(v) > 1:
			pdf = np.zeros((100,100))
			length = len(pdf_list)
			for a in v:
				pdf += (1/length)*pdf_list[a]
			# scalarize_minmax([pdf_list[a] for a in v],problem.s0[k*3:k*3+3],problem.nA)
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
	# pdb.set_trace()

	#Start the tree with the root node being [], blank assignment
	root = Node(None, [], [], np.inf, np.inf, [], None)
	print("Added root node")

	maps_assigned = np.zeros(n_obj)
	explore_node = [root]

	for i in range(n_agents):
		print("Looking at assignments for agent: ", i)

		new_explore_node = []
		for curr_node in explore_node:
			alloc_comb = generate_alloc_nodes(root,curr_node,n_obj,n_agents)
			print("Alloc_comb: ", alloc_comb)
			# #pd.set_trace()
			for a in alloc_comb:
				print("Looking at next assignment for node: ", i)
				print("Parent assignment: ", curr_node.tasks)
				print("Assignment: ", a)
				print("Nodes pruned so far: ", nodes_pruned)
				# pdb.set_trace()
				node = Node(i, a, [], np.inf, np.inf, [], curr_node)
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
					# #pd.set_trace()
					if erg > upper:
						node.alive = False
						prune = True
						print("Don't explore further")
						# pdb.set_trace()
						nodes_pruned += 1 
						break
					node.indv_erg.append(erg)
				if not prune:
					# print("Not pruning this node")
					node.tasks = a
					curr_node.children.append(node)
					# print("node.indv_erg: ", node.indv_erg)
					new_explore_node.append(node)
		explore_node = new_explore_node

		# if i == n_agents-1:
		# 	print("Final agent assignments have been checked!")
		# 	for e in explore_node:
		# 		print(e.tasks)
	# final_allocation = find_best_allocation(root)

	print("Number of nodes pruned: ", nodes_pruned)
	total_alloc = len(generate_allocations(n_obj,n_agents))
	print("Total number of nodes: ", total_alloc*n_agents)
	print("Percentage of nodes pruned: ", nodes_pruned/total_alloc)
	return root, problem.s0

def branch_and_bound_main(pbm_file,n_agents,start_pos=[-1]):
	# pbm_file = "random_map_1.pickle"
	# n_agents = 2

	print("file: ", pbm_file)
	root, start_pos = branch_and_bound(pbm_file,n_agents,start_pos)

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
	
	# print("Final allocation: ", final_allocation)
	return best_alloc,indv_erg[min_idx],start_pos

if __name__ == "__main__":
	dx = [-2,-2,-2,-2,-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
	dy = [-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2]
	all_exp = []
	cnt = 0
	for file in os.listdir("./build_prob/instances/"):
		print("filename: ", file)

		map_start_perturb = {}
		if not file.startswith("random"):
			continue
		n_agents = 2
		best_alloc_OG, indv_erg_OG, start_pos_OG = branch_and_bound_main(file,n_agents)
		print("best_alloc_OG: ", best_alloc_OG)
		print("indv_erg_OG: ", indv_erg_OG)
		print("start_pos_OG: ", start_pos_OG)
		# pdb.set_trace()
		map_start_perturb["0_0_0_0"] = (best_alloc_OG,indv_erg_OG)
		print("map_start_perturb: ", map_start_perturb)
		np.save(file.split(".")[0]+"_perturbation.npy",map_start_perturb)
		# pdb.set_trace()

		for i in range(len(dx)):
			start_pos = copy.deepcopy(start_pos_OG)
			start_pos[0] += dx[i]/100
			start_pos[1] += dy[i]/100
			print("dx,dy: ", dx[i],dy[i])
			print("start_pos: ", start_pos)
			# pdb.set_trace()
			if start_pos[0] >= 0 and start_pos[0] <= 1 and start_pos[1] >= 0 and start_pos[1] <= 1:
				best_alloc, indv_erg, _ = branch_and_bound_main(file,n_agents,start_pos)
				print("best_alloc: ", best_alloc)
				print("indv_erg: ", indv_erg)
				print("start_pos: ", start_pos)
				map_start_perturb[str(dx[i])+"_"+str(dy[i])+"0_0"] = (best_alloc,indv_erg)
				print("map_start_perturb: ", map_start_perturb)
				# pdb.set_trace()
			else:
				print("skipping this start position!")

			start_pos = copy.deepcopy(start_pos_OG)
			start_pos[3] += dx[i]/100
			start_pos[4] += dy[i]/100
			print("dx,dy: ", dx[i],dy[i])
			print("start_pos: ", start_pos)
			# pdb.set_trace()

			if start_pos[3] >= 0 and start_pos[3] <= 1 and start_pos[4] >= 0 and start_pos[4] <= 1:
				best_alloc, indv_erg, _ = branch_and_bound_main(file,n_agents,start_pos)
				print("best_alloc: ", best_alloc)
				print("indv_erg: ", indv_erg)
				print("start_pos: ", start_pos)
				map_start_perturb["0_0_"+str(dx[i])+"_"+str(dy[i])] = (best_alloc,indv_erg)
				print("map_start_perturb: ", map_start_perturb)
				# pdb.set_trace()
			else:
				print("Skipping this start position!")

		np.save(file.split(".")[0]+"_perturbation_correct.npy",map_start_perturb)
		print("outside while loop")

		
		all_exp.append(map_start_perturb)
		# break

	print("All experiment results: ", all_exp)
	np.save("perturbation_experiment.npy",all_exp)














