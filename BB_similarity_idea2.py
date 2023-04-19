import matplotlib.pyplot as plt
import numpy as np

import common
import scalarize
from ergodic_coverage import ErgCover
import jax.numpy as jnp
import pdb
import ergodic_metric

import time
from utils import *
from explicit_allocation import *

np.random.seed(101)

""""
First find an incumbent solution using greedy approach to get an "upperd_bound". The incumbent solution
will be the max(indv_ergodicities) for the greedy allocation combination

Initialize a "dominant" set to []. This will house the definitely bad assignments (not entire allocation)

The root node is [], i.e. no assignment has been done so far
Then each level corresponds to the assignment of each agent to the maps

For each node, if the the max(erg) so far is > upper bound then discard that node

Update the upper bound in two ways:
1. Randomly fathom a node/choose a promising node (using EEE function) to fathom (i.e, fix the allocation so far and perform greedy allocation
again to check if upper bound can be updated)
2. When the last level of the tree is reached, a complete solution is obtained and upper bound can be updated

**************************
within each node, can we calculate the ergodicity of the agent on the map combination we expect it to have the worst 
performance on? This would be the second level of heuristic
***************************

****************************
Similarity Idea 2:
Once the ergodicity of the agent on map 1, 2..n individually is calculated. Estimate the ergodicities when the agent is assigned
say map 1 and 2. If 1 and 2 are similar, then the individual erg shouldn't change so much. Else it will increase a lot and so
can possibly be pruned.
***************************


Continue this till the leaf nodes are reached. The number of levels in the tree will be equal to the
 number of agents.


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
		self.max_erg = 0

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

def greedy_alloc(problem, n_agents,node = None, sensor_footprint = 15):
	#Allocate based on information in a window centered around the agent

	n_obj = len(problem.pdfs)
	agents_allotted = []
	maps_allotted = []
	if node:
		current_node = node
		while current_node.parent:
			agents_allotted.append(current_node.agent)
			maps_allotted = maps_allotted + list(current_node.tasks)
			current_node = current_node.parent

	# print("\nAgents already allotted: ", agents_allotted)
	# print("\nMaps already assigned: ", maps_allotted)

	if len(agents_allotted) == n_agents:
		print("all agents are allotted")
		return -1

	agent_locs = []
	for i in range(n_agents):
		if i in agents_allotted:
			agent_locs.append((-1,-1))
		else:
			agent_locs.append((round(problem.s0[0+i*3]*100),round(problem.s0[1+i*3]*100)))

	agent_scores = np.zeros((n_agents,n_obj))

	# Calculate how much information agent i gets when allotted map p

	for i in range(n_agents):
		if i in agents_allotted:
			continue

		x_min = max(agent_locs[i][0]-sensor_footprint,0)
		x_max = min(agent_locs[i][0]+sensor_footprint,100)

		y_min = max(agent_locs[i][1]-sensor_footprint,0)
		y_max = min(agent_locs[i][1]+sensor_footprint,100)

		xr = np.arange(x_min,x_max)
		yr = np.arange(y_min,y_max)

		for p in range(n_obj):
			if p in maps_allotted:
				continue
			for x in xr:
				for y in yr:
					agent_scores[i][p] += problem.pdfs[p][y][x]

	# print("Agent scores: ", agent_scores)

	maps_assigned = np.zeros(n_obj)
	allocation = {}

	if n_agents > n_obj:
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
		print("No >= Na")
		agents_assigned = np.zeros(n_agents)
		
		for j in range(n_agents):
			allocation[j] = []
		
		for i in range(n_obj):
			# print("Checking for map: ", i)
			if i in maps_allotted:
				continue
			k = 0

			# Sort the amount of information on a map for different agents in descending order

			map_scores = [a[i] for a in agent_scores]
			info = sorted(map_scores, reverse=True) 
			found = False
			while not found:
				idx = map_scores.index(info[k]) # Free agent with max info on this map

				#Capping the maximum number of maps that can be assigned to an agent to ensure no agent is left without maps
				if (agents_assigned[idx] > 0 and np.count_nonzero(agents_assigned == 0) == n_obj - sum(agents_assigned)) or idx in agents_allotted:
					# print("Agent is already assigned max possible maps")
					k += 1
				else:
					allocation[idx] = allocation[idx] + [i]
					found = True
					agents_assigned[idx] += 1
					# print("allocation so far: ", allocation)
	print("Incumbent allocation: ", allocation)
	return allocation

def generate_alloc_nodes(curr_node,n_obj,n_agents):
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
	if temp.depth+1 == n_agents:
		start = len(maps_left)
	else:
		start = 1

	for n in np.arange(start,len(maps_left)-(n_agents-temp.depth-1)+1):
		assignments = assignments + list(itertools.combinations(maps_left, n))
	
	return assignments

def traverse(node, result=[], path =[]):
	path.append((node.tasks,node.indv_erg))
	if len(node.children)==0:
		print(path)
		result.append(path.copy())
		print("result: ", result)
		path.pop()
	else:
		for child in node.children:
			traverse(child,result)
		path.pop()
	return result

def update_upper(node,upper):
	indv_erg = []
	temp = node
	while(temp):
		print("\nAgent: ", temp.agent)
		print("\nAssignment: ", temp.tasks)
		print("\nIndv erg: ", temp.indv_erg)
		indv_erg = indv_erg + temp.indv_erg
		print("\nindv erg all: ", indv_erg)
		temp = temp.parent
		# pdb.set_trace()
	print("The indv ergodicities of the entire allocations: ", indv_erg)
	if upper > max(indv_erg):
		print("***updating upper bound")
		upper = max(indv_erg)
	return upper

def find_best_allocation(root,values,alloc,indv_erg):
	# list to store path
	if root == None:
		return 
	if len(root.children) == 0:
		# print("Reached leaf node")
		path = {}
		max_erg = []
		values.append(root)
		path[values[0].agent] = values[0].tasks
		max_erg += values[0].indv_erg
		# print("Path before adding all elements in values: ", path)
		# print("values: ", values)
		for i in np.arange(1,len(values)):
			path[values[i].agent] = values[i].tasks
			max_erg += values[i].indv_erg
		alloc.append(path)
		indv_erg.append(max_erg)
		# print("Path found: ", path)
		values.pop()
		return
	# print(str(root.agent)+" has "+str(len(root.children))+" children!")
	values.append(root)
	# print("Values: ", values)
	for child in root.children:
		find_best_allocation(child,values,alloc,indv_erg)
	values.pop()

def branch_and_bound(pbm_file, n_agents, n_scalar, start_pos, random_start=True, scalarize=False):

	start_time = time.time()
	pbm_file_complete = "./build_prob/random_maps/" + pbm_file
	problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)

	n_obj = len(problem.pdfs)
	problem.nA = 100

	if n_obj < n_agents or n_obj > 6:
		print("No < Na")
		return [],0,0,0

	#Generate random starting positions or read from file
	if random_start:
		pos = np.random.uniform(0,1,2*n_agents)
		problem.s0 = []
		k = 0
		for i in range(n_agents):
			problem.s0.append(pos[k])
			problem.s0.append(pos[k+1])
			problem.s0.append(0)
			k += 2
		problem.s0 = np.array(problem.s0)
	else:
		problem.s0 = start_pos.item().get(pbm_file)

	# display_map(problem,problem.s0,window=15)

	pdf_list = problem.pdfs
	
	print("Agent start positions allotted!")

	#Generate incumbent solution using Greedy approach
	incumbent = greedy_alloc(problem,n_agents)
	incumbent_erg = np.zeros(n_obj)

	nodes_pruned = 0
	nodes_explored = 0

	#Find the upper bound for the incumbent solution
	for k,v in incumbent.items():
		if len(v) > 1:
			if scalarize:
				pdf = scalarize_minmax([pdf_list[a] for a in v],problem.s0[k*3:k*3+3],problem.nA)
			else:
				pdf = np.zeros((100,100))
				for a in v:
					pdf += (1/len(v))*pdf_list[a]
		else:
			pdf = pdf_list[v[0]]

		pdf = jnp.asarray(pdf.flatten())
		
		# Just run ergodicity optimization for fixed iterations to get ergodic trajectory 
		control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)
		
		# Calculate individual ergodicities using the gotten trajectory
		for p in v:
			pdf_indv = jnp.asarray(pdf_list[p].flatten())
			EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
			incumbent_erg[p] = EC.fourier_ergodic_loss(control,problem.s0[3*k:3+3*k])

	upper = max(incumbent_erg)
	print("Incumbent allocation: ", incumbent)
	print("Incumber Ergodicities: ", incumbent_erg)
	print("Initial Upper: ", upper)
	#pdb.set_trace()

	#Start the tree with the root node being [], blank assignment
	root = Node(None, [], [], np.inf, np.inf, [], None)
	print("Added root node")

	# Nodes that are alive or not pruned
	explore_node = [root]
	promise_nodes = np.zeros(n_agents)

	for i in range(n_agents):
		print("Looking at assignments for agent: ", i)
		agent_map_erg = {}
		#pbd.set_trace()
		new_explore_node = []
		for curr_node in explore_node:
			alloc_comb = generate_alloc_nodes(curr_node,n_obj,n_agents)
			print("Alloc_comb: ", alloc_comb)
			# pdb.set_trace()
			for a in alloc_comb:
				print("Looking at next possible assignment for agent: ", i)
				print("Parent Node allocation: ", curr_node.tasks)
				
				node = Node(i, a, [], np.inf, np.inf, [], curr_node)
				print("Alloc assigned: ", a)
				
				prune = False

				if a not in agent_map_erg.keys():
					agent_map_erg[a] = []
					if len(a) > 1:
						if scalarize:
							pdf = scalarize_minmax([pdf_list[j] for j in a],problem.s0[i*3:i*3+3],problem.nA)
						else:
							pdf = np.zeros((100,100))
							for j in a:
								pdf += (1/len(a))*pdf_list[j]
					else:
						pdf = pdf_list[a[0]]

					og_pdf = pdf
					pdf = jnp.asarray(pdf.flatten())

					#Just run ergodicity optimization for fixed iterations and see which agent achieves best ergodicity in that time
					control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*i:3+3*i], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)

					#Can have a heuristic to show the order in which to evaluate the indv ergodicities
					for p in a:
						pdf_indv = jnp.asarray(pdf_list[p].flatten())
						EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
						erg = EC.fourier_ergodic_loss(control,problem.s0[3*i:3+3*i])
						agent_map_erg[a].append(erg)
						print("erg: ", erg)
						# #pd.set_trace()
						if erg > upper:
							node.alive = False
							prune = True
							print("Don't explore further")
							nodes_pruned += 1 
							continue
						node.indv_erg.append(erg)
						curr_node.children.append(node)
						print("node.indv_erg: ", node.indv_erg)
						# if erg:
						# 	node.max_erg = max(erg)
				else:
					print("\nAlready saw this allocation!")
					# pdb.set_trace()
					for e in agent_map_erg[a]:
						if e > upper:
							node.alive = False
							prune = True
							print("Don't explore further")
							nodes_pruned += 1 
							continue
						node.indv_erg.append(e)
						curr_node.children.append(node)
						print("node.indv_erg: ", node.indv_erg)

				if node.depth == n_agents:
					nodes_explored += 1
					if(node.alive):
						print("\n******Trying to update the upper bound!")
						print("\nparent agent: ", node.parent.agent)
						print("\ncurre node agent: ", curr_node.agent)
						# pdb.set_trace()
						upper = update_upper(node,upper)
				if not prune:
					print("Not pruning this node")
					new_explore_node.append(node)
					fathom = np.random.randint(1,10)
					score = H_function(og_pdf,problem.s0[3*i:3*i+3]) #Find how promising a node is and fathom accordingly
					if score > promise_nodes[i] and node.depth < n_agents:
					# if fathom > 7 and node.depth < n_agents:
						# promise_nodes[i] = score
						print("\nrandom fathom of node!!")
						print("\nDepth of node: ", node.depth)
						# pdb.set_trace()
						new_incumbent = greedy_alloc(problem,n_agents,node)
						new_incumbent_erg = np.zeros(n_obj)
						print("\nGot incumbent allocation!")

						#Find the upper bound for the incumbent solution
						for k,v in new_incumbent.items():
							if len(v) > 1:
								if scalarize:
									pdf = scalarize_minmax([pdf_list[a] for a in v],problem.s0[k*3:k*3+3],problem.nA)
								else:
									pdf = np.zeros((100,100))
									for a in v:
										pdf += (1/len(v))*pdf_list[a]
							elif len(v) == 0:
								continue
							else:
								pdf = pdf_list[v[0]]

							pdf = jnp.asarray(pdf.flatten())
							
							#Just run ergodicity optimization for fixed iterations and see which agent achieves best ergodicity in that time
							control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)
							
							for p in v:
								pdf_indv = jnp.asarray(pdf_list[p].flatten())
								EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
								new_incumbent_erg[p] = EC.fourier_ergodic_loss(control,problem.s0[3*k:3+3*k])

						if upper > max(new_incumbent_erg):
							print("\n************Updating upper**************")
							# pdb.set_trace()
							upper = max(new_incumbent_erg)
				# #pd.set_trace()
		explore_node = new_explore_node

	print("Number of nodes pruned: ", nodes_pruned)
	print("Number of nodes explored: ", nodes_explored)
	alloc_comb = generate_allocations(n_obj,n_agents)
	print("Total number of leaf nodes: ", len(alloc_comb))
	per_leaf_prunes = (len(alloc_comb) - nodes_explored)/len(alloc_comb)
	print("percentage of leaf nodes pruned: ", per_leaf_prunes)
	# pdb.set_trace()

	# result = []
	# path = []
	# traverse(root,result,path)
	# print("All paths found: ", result)

	# min_max = np.ones(len(result))*100

	# for idx,p in enumerate(result):
	# 	erg = []
	# 	print("len: ", len(p))
	# 	print(p)
	# 	# pdb.set_trace()
	# 	if len(p) < n_agents:
	# 		continue
	# 	for node in p:
	# 		erg = erg + list(node[1])
	# 	print("erg: ", erg)
	# 	min_max[idx] = min(erg)
	# print("minmax: ", min_max)
	# best_path =  np.argmin(min_max)
	# print("Best path index: ", best_path)
	# print("Best path: ", result[best_path])

	# best_alloc = {}
	# for idx,b in enumerate(result[best_path]):
	# 	best_alloc[idx] = b[0]

	# # title = str(result[best_path][0][0]) + str(result[best_path][1][0])
	# # pdb.set_trace()

	# display_map(problem,problem.s0,pbm_file,title="Best Allocation: "+str(best_alloc))
	# runtime = time.time() - start_time

	values = []
	alloc = []
	indv_erg = []
	find_best_allocation(root,values,alloc,indv_erg)
	print("All paths found: ", alloc)
	print("Individual ergodicities: ", indv_erg)
	print("Number of agents: ", n_agents)
	# pdb.set_trace()

	#Among all the allocations found, pick the one with min max erg
	max_erg = []
	for idx,e in enumerate(indv_erg):
		# print("e: ", e)
		# print("len(e): ", len(e))
		if len(alloc[idx]) < n_agents+1:
			max_erg.append(100)
		else:
			max_erg.append(max(e))

	print("Max ergodicities: ", max_erg)
	min_idx = np.argmin(max_erg)

	best_alloc = alloc[min_idx]

	print("The best allocation according to minmax metric: ", best_alloc)
	# pdb.set_trace()
	# print("Final allocation: ", final_allocation)
	runtime = time.time() - start_time
	return best_alloc,runtime,per_leaf_prunes,indv_erg[min_idx]

	# return best_alloc, runtime, per_leaf_prunes

if __name__ == "__main__":
	pbm_file = "4_maps_example_3.pickle"
	n_agents = 2
	n_scalar = 10
	final_allocation, runtime, per_leaf_prunes = branch_and_bound(pbm_file,n_agents,n_scalar)
	print("Final allocation: ", final_allocation)
	print("Runtime: ", runtime)
	print("per pruned: ", per_leaf_prunes)

