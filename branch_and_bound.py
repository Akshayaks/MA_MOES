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

**************************
within each node, can we calculate the ergodicity of the agent on the map combination we expect it to 
have the worst performance on?
This would be the second level of heuristic
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

def greedy_alloc(problem, n_agents, sensor_footprint=15,node = None):
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

	# print("\nAgents already allotted: ", agents_allotted)
	# print("\nMaps already assigned: ", maps_allotted)
	if len(agents_allotted) == n_agents:
		# print("all agents are allotted")
		return -1

	agent_locs = []
	for i in range(n_agents):
		if i in agents_allotted:
			agent_locs.append((-1,-1))
		else:
			agent_locs.append((round(problem.s0[0+i*3]*100),round(problem.s0[1+i*3]*100)))
	# print("Agent locations: ", agent_locs)

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

	#Calculate how much information agent 1 and agent 2 can cover when allocatted to map1 and map2 respectively and vice versa

	for i in range(n_agents):
		if i in agents_allotted:
			continue
		xr = np.arange(x_range[i][0],x_range[i][1])
		yr = np.arange(y_range[i][0],y_range[i][1])
		# print("agent: ", i)

		for p in range(n_obj):
			if p in maps_allotted:
				continue
			# print("objective: ", p)
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
		# print("Allocation initialized to empty list")
		for i in range(n_obj):
			# print("Checking for map: ", i)
			if i in maps_allotted:
				continue
			k = 0

			#Sort the amount of information on a map for different agents in descending order
			# print("Agent scores for chosen map: ", [a[i] for a in agent_scores])
			info = sorted([a[i] for a in agent_scores], reverse=True) 
			found = False
			while not found:
				idx = [a[i] for a in agent_scores].index(info[k])
				# print("Agent with max info : ", idx)

				#Capping the maximum number of maps that can be assigned to an agent to ensure to agent is left without maps
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

def branch_and_bound(pbm_file, n_agents, n_scalar, random_start=True, start_pos_file="", scalarize=False):

	start_time = time.time()
	pbm_file_complete = "./build_prob/test_cases/" + pbm_file
	problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)
	n_obj = len(problem.pdfs)
	problem.nA = 100 
	# nA = problem.nA

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
		start_pos = np.load(start_pos_file,allow_pickle=True)
		problem.s0 = start_pos.item().get(pbm_file)

	# display_map(problem,problem.s0,window=15)

	pdf_list = problem.pdfs
	
	print("Agent start positions allotted!")

	#Generate incumbent solution using Greedy approach
	incumbent = greedy_alloc(problem,n_agents,n_scalar)
	incumbent_erg = np.zeros(n_obj)

	final_allocation = {}

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
	#pdb.set_trace()

	#Start the tree with the root node being [], blank assignment
	root = Node(None, [], [], np.inf, np.inf, [], None)
	print("Added root node")

	# Nodes that are alive
	explore_node = [root]

	for i in range(n_agents):
		print("Looking at assignments for agent: ", i)
		#pd.set_trace()
		new_explore_node = []
		for curr_node in explore_node:
			alloc_comb = generate_alloc_nodes(curr_node,n_obj,n_agents)
			print("Alloc_comb: ", alloc_comb)
			#pdb.set_trace()
			for a in alloc_comb:
				print("Looking at next possible assignment for agent: ", i)
				print("curr_node: ", curr_node)
				# print("Node created: ", Node(i, a, [], np.inf, np.inf, [], curr_node))
				# #pd.set_trace()
				node = Node(i, a, [], np.inf, np.inf, [], curr_node)
				print("Alloc assigned: ", a)
				# #pd.set_trace()
				prune = False
				if len(a) > 1:
					if scalarize:
						pdf = scalarize_minmax([pdf_list[j] for j in a],problem.s0[i*3:i*3+3],problem.nA)
					else:
						pdf = np.zeros((100,100))
						for j in a:
							pdf += (1/len(a))*pdf_list[j]
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
					# if erg:
					# 	node.max_erg = max(erg)
				if not prune:
					nodes_explored += 1
					print("Not pruning this node")
					# #pd.set_trace()
					curr_node.children.append(node)
					print("node.indv_erg: ", node.indv_erg)
					new_explore_node.append(node)
				# #pd.set_trace()
		explore_node = new_explore_node

	print("Number of nodes pruned: ", nodes_pruned)
	print("Number of nodes explored: ", nodes_explored)
	print("Total number of nodes: ", nodes_explored + nodes_pruned)
	print("Percentage of nodes pruned: ", nodes_pruned/(nodes_pruned + nodes_explored))

	result = []
	path = []
	traverse(root,result,path)
	print("All paths found: ", result)

	min_max = np.zeros(len(result))

	for idx,p in enumerate(result):
		erg = []
		for node in p:
			erg = erg + list(node[1])
		print("erg: ", erg)
		min_max[idx] = min(erg)
	print("minmax: ", min_max)
	best_path =  np.argmin(min_max)
	print("Best path index: ", best_path)
	print("Best path: ", result[best_path])

	return root, problem.s0

if __name__ == "__main__":
	pbm_file = "3_maps_example_3.pickle"
	n_agents = 2
	n_scalar = 10
	final_allocation = branch_and_bound(pbm_file,n_agents,n_scalar)
	print("Final allocation: ", final_allocation)


