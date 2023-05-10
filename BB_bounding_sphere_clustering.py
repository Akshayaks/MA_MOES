import numpy as np
import os
import common
from ergodic_coverage import ErgCover
import ergodic_metric
from utils import *
from explicit_allocation import *
import time
import matplotlib.pyplot as plt

from miniball import miniball

np.random.seed(100)

"""
Branch and bound using clusters of maps based on the norm of the difference in their weighted fourier 
coefficients. The agents are then allotted to one cluster instead of being allotted to a set
of maps. Every level of the tree will correspond to the assignment of an agent to one cluster.
Thus the tree will have a depth equal to the number of agents.

This code needs to be optimized for ignoring bad partial allocations like in BB_optimized
"""

class ClusteringNode:
    def __init__(self, tasks, radius, children, parent):
        self.tasks = tasks
        self.radius = radius          #The radius of the cluster
        self.children = children    #List of the children of the Node
        self.parent = parent
        self.alive = True
        if parent:
            self.depth = parent.depth+1
        else:
            self.depth = 0 

    def kill(self):
        self.alive = False    #Pruning/bounding

class Node:
	def __init__(self):
		self.agent = None
		self.tasks = []
		self.cluster = -1
		self.indv_erg = []   #List of indv ergodicities on assigned maps
		self.upper = None    #The maximum of the indv ergodicities so far
		self.children = []   #List of the children of the Node
		self.parent = None
		self.alive = True    #By default the node is alive (need to be explored)
		self.depth = 0

	def __init__(self, agent, tasks, cluster, indv_erg, upper, U2, children, parent):
		self.agent = agent
		self.tasks = tasks
		self.cluster = cluster
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

def generate_alloc_nodes(root,curr_node,num_agents,clusters):
	clusters_assigned = []
	temp = curr_node
	while(curr_node):
		clusters_assigned = clusters_assigned + list(curr_node.cluster)
		curr_node = curr_node.parent
	clusters_left = []
	for i in range(len(clusters)):
		if i not in clusters_assigned:
			clusters_left.append(i)
	assignments = []
	if temp.depth+1 == n_agents:
		start = len(clusters_left)
	else:
		start = 1

	for n in np.arange(start,len(clusters_left)-(n_agents-temp.depth-1)+1):
		assignments = assignments + list(itertools.combinations(clusters_left, n))

	return assignments

def find_best_allocation(root,values,alloc,indv_erg,n_agents,n_obj):
	if root == None:
		return 
	if len(root.children) == 0:
		path = {}
		max_erg = []
		values.append(root)
		path[values[0].agent] = values[0].tasks
		max_erg += values[0].indv_erg
		for i in np.arange(1,len(values)):
			path[values[i].agent] = values[i].tasks
			max_erg += values[i].indv_erg
		if len(path) == n_agents+1:
			alloc.append(path)
			indv_erg.append(max_erg)
		else:
			print("Incomplete path")
		values.pop()
		return
	values.append(root)
	for child in root.children:
		find_best_allocation(child,values,alloc,indv_erg,n_agents,n_obj)
	values.pop()

def update_upper(node,upper):
	indv_erg = []
	temp = node
	while(temp):
		indv_erg = indv_erg + temp.indv_erg
		temp = temp.parent
	if upper > max(indv_erg):
		print("***updating upper bound")
		upper = max(indv_erg)
	return upper

def greedy_alloc(problem, clusters, n_agents, n_scalar, node = None):
	#Allocate based on a circle around the agent and calculating the information in that region

	n_obj = len(problem.pdfs)
	agents_allotted = []
	clusters_allotted = []
	if node:
		current_node = node
		while current_node.parent:
			agents_allotted.append(current_node.agent)
			clusters_allotted = clusters_allotted + list(current_node.cluster)
			current_node = current_node.parent
	if len(agents_allotted) == n_agents:
		print("all agents are allotted")
		return -1

	sensor_footprint = 15
	agent_locs = []
	for i in range(n_agents):
		if i in agents_allotted:
			agent_locs.append((-1,-1))
		else:
			agent_locs.append((round(problem.s0[0+i*3]*100),round(problem.s0[1+i*3]*100)))

	x_range = []
	y_range = []

	for i in range(n_agents):
		if i in agents_allotted:
			x_range.append((-1,-1))
			y_range.append((-1,-1))
		else:
			x_range.append((max(agent_locs[i][0]-sensor_footprint,0),min(agent_locs[i][0]+sensor_footprint,100)))
			y_range.append((max(agent_locs[i][1]-sensor_footprint,0),min(agent_locs[i][1]+sensor_footprint,100)))

	agent_scores = np.zeros((n_agents,len(clusters)))

	#Calculate how much information each agent can get when allocatted to each map

	for i in range(n_agents):
		if i in agents_allotted:
			continue
		xr = np.arange(x_range[i][0],x_range[i][1])
		yr = np.arange(y_range[i][0],y_range[i][1])

		for p in range(len(clusters)):
			if p in clusters_allotted:
				continue
			for x in xr:
				for y in yr:
					for mapi in clusters[p]:
						agent_scores[i][p] += problem.pdfs[mapi][y][x]
	clusters_assigned = np.zeros(len(clusters))
	allocation = {}

	if n_agents == len(clusters):
		print("No = Na")
		for i in range(n_agents):
			k = 0
			erg = sorted(agent_scores[i][:], reverse=True)
			found = False
			while not found:
				idx = agent_scores[i].tolist().index(erg[k])
				if clusters_assigned[idx]:
					k += 1
				else:
					allocation[i] = [idx]
					found = True
					clusters_assigned[idx] = 1
			print("\nClusters assigned: ", clusters_assigned) 
	else:
		print("No > Na")
		agents_assigned = np.zeros(n_agents)
		for j in range(n_agents):
			allocation[j] = []
		for i in range(len(clusters)):
			if i in clusters_allotted:
				continue
			k = 0
			erg = sorted([a[i] for a in agent_scores], reverse=True)
			found = False
			while not found:
				idx = [a[i] for a in agent_scores].index(erg[k])
				zero_map_agents = list(agents_assigned).count(0)
				if (agents_assigned[idx] > 0 and len(clusters)-i == zero_map_agents) or idx in agents_allotted:
					k += 1
				else:
					allocation[idx] = allocation[idx] + [i]
					found = True
					agents_assigned[idx] += 1
	print("The final allocations are as follows: ", allocation)
	return allocation

def get_subsets(s):
    subsets = []
    for i in range(1,len(s)):
        subsets = subsets + list(itertools.combinations(s, i))
    return subsets

def branch_and_bound(problem, clusters, n_agents, start=[-1], scalarize=False):
	start_time = time.time()
	n_scalar = 10
	n_obj = len(problem.pdfs)

	pdf_list = problem.pdfs
	problem.nA = 100

	#Generate incumbent solution using Greedy approach
	incumbent = greedy_alloc(problem,clusters,n_agents,n_scalar)
	print("\nGot incumbent cluster allocation!")

	incumbent_erg = np.zeros(n_obj)

	nodes_pruned = 0

	#Find the upper bound for the incumbent solution
	for k,v in incumbent.items():
		pdf = np.zeros((100,100))
		n_maps = 0
		for a in v:
			for mapi in clusters[a]:
				pdf += pdf_list[mapi]
				n_maps += 1
		pdf = (1/n_maps)*pdf
		pdf = np.asarray(pdf.flatten())
		
		#Just run ergodicity optimization for fixed iterations and see which agent achieves best ergodicity in that time
		control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)
		
		for p in v:
			for mapi in clusters[p]:
				pdf_indv = np.asarray(pdf_list[mapi].flatten())
				EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
				incumbent_erg[mapi] = EC.fourier_ergodic_loss(control,problem.s0[3*k:3+3*k])

	upper = max(incumbent_erg)
	print("Incumbent allocation: ", incumbent)
	print("Incumber Ergodicities: ", incumbent_erg)
	print("Initial Upper: ", upper)

	#Start the tree with the root node being [], blank assignment
	nodes_explored = 0
	root = Node(None, [], [], [], np.inf, np.inf, [], None)
	explore_node = [root]
	agent_alloc_pruned = [[] for _ in range(n_agents)]

	for i in range(n_agents):
		print("Looking for allocations for agent: ", i)
		agent_cluster_erg = {}
		new_explore_node = []
		for curr_node in explore_node:
			alloc_cluster = generate_alloc_nodes(root,curr_node,n_agents,clusters)
			# print("Generated allocations: ", alloc_cluster)
			for a in alloc_cluster:
				# print("a: ", a)
				am = []
				for ai in a:
					# print("ai: ", ai)
					# print("clusters[ai]: ", clusters[ai])
					# am = am + clusters[ai]
					am = np.concatenate((am,clusters[ai]),axis=None)
				# 	print("am: ", am)
				# print("Maps assigned to the agent: ", am)
				# breakpoint()
				node = Node(i, am, a, [], np.inf, np.inf, [], curr_node)
				prune = False
				bad_alloc = False

				if a not in agent_cluster_erg.keys():
					subsets = get_subsets(list(a))
					for s in subsets:
						if s in agent_alloc_pruned[i]:
							print("\n**************************Alloc contains subset of bad information map!")
							agent_alloc_pruned[i].append(a)
							node.alive = False
							prune = True
							nodes_pruned += 1 
							bad_alloc = True
							break
					if bad_alloc:
						continue
					# print("Not a bad alloc")
					agent_cluster_erg[a] = []

					pdf = np.zeros((100,100))
					n_maps = len(am)
					for ai in a:
						for mapi in clusters[ai]:
							pdf += (1/n_maps)*pdf_list[mapi]
					
					pdf = np.asarray(pdf.flatten())

					#Just run ergodicity optimization for fixed iterations and see which agent achieves best ergodicity in that time
					control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*i:3+3*i], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)

					for p in a:
						for mapi in clusters[p]:
							pdf_indv = np.asarray(pdf_list[mapi].flatten())
							EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
							erg = EC.fourier_ergodic_loss(control,problem.s0[3*i:3+3*i])
							agent_cluster_erg[a].append(erg)
							if erg > upper:
								node.alive = False
								prune = True
								# print("Don't explore further")
								nodes_pruned += 1 
								agent_alloc_pruned[i].append(a)
								break
							node.indv_erg.append(erg)
						if prune:
							break
				else:
					# print("\nAlready saw this allocation!")
					for e in agent_cluster_erg[a]:
						if e > upper:
							node.alive = False
							prune = True
							# print("Don't explore further")
							nodes_pruned += 1 
							break
						node.indv_erg.append(e)
				if node.depth == n_agents:
					nodes_explored += 1
					if(node.alive):
						upper = update_upper(node,upper)
				if not prune:
					# print("Not pruning this node")
					node.cluster = a
					node.tasks = am
					curr_node.children.append(node)
					new_explore_node.append(node)
		explore_node = new_explore_node

	print("Number of nodes pruned: ", nodes_pruned)
	total_alloc = len(generate_allocations(n_obj,n_agents))
	print("Total number of nodes: ", total_alloc*n_agents)
	print("Percentage of nodes pruned: ", nodes_pruned/total_alloc)
	return root, problem.s0

def find_minmax(indv_erg):
	sorted_erg = [sorted(x,reverse=True) for x in indv_erg]
	n_obj = len(sorted_erg[0])
	col_with_index = []
	col = []
	tied_rows = np.arange(0,len(sorted_erg))
	for i in range(n_obj):
		col = []
		col_with_index = []
		for tied_idx in tied_rows:
			col_with_index.append((sorted_erg[tied_idx],tied_idx))
			col.append(sorted_erg[tied_idx])
		min_idx = np.argmin([c[i] for c in col])
		min_idx_sorted_array = col_with_index[min_idx][1]
		min_val = np.min([c[i] for c in col])
		num = 0
		ties_at = []
		for j in range(len(col_with_index)):
			if col_with_index[j][0][i] == min_val:
				num += 1
				ties_at.append(col_with_index[j][1])
		if num == 1 or num == 0:
			break
		else:
			tied_rows = ties_at
	return min_idx_sorted_array


def branch_and_bound_main(pbm,clusters,n_agents,start_pos=[-1]):
	start_time = time.time()
	root, start_pos = branch_and_bound(pbm,clusters,n_agents,start_pos)

	values = []
	alloc = []
	indv_erg = []
	find_best_allocation(root,values,alloc,indv_erg,n_agents,len(pbm.pdfs))
	print("All paths found: ", alloc)
	print("Individual ergodicities: ", indv_erg)
	print("Number of agents: ", n_agents)
	print("Number of clusters: ", clusters)
	min_idx = find_minmax(indv_erg)
	best_alloc = alloc[min_idx]

	print("The best allocation according to minmax metric: ", best_alloc)
	runtime = time.time() - start_time
	return best_alloc,indv_erg[min_idx],start_pos,runtime

def get_minimal_bounding_sphere(pdf_list,nA,pix):
    FC = []
    for pdf in pdf_list:
        EC = ergodic_metric.ErgCalc(pdf.flatten(),1,nA,10,pix)
        FC.append(EC.phik*np.sqrt(EC.lamk))
    res = miniball(np.asarray(FC,dtype=np.double))
    pdf_FC = res["center"]
    # pdf_FC = np.divide(res["center"],np.sqrt(EC.lamk))
    minmax = res["radius"]
    # print(res['radius'])
    # breakpoint()

    return pdf_FC, minmax

def generate_alloc_nodes_for_clustering(curr_node,n_obj,n_agents):
    maps_assigned = []
    temp = curr_node
    while(curr_node):
        maps_assigned = maps_assigned + list(curr_node.tasks)
        curr_node = curr_node.parent
    # print("Maps assigned: ", maps_assigned)
    maps_left = [] #list(set(np.arange(1,n_obj+1)) - set(maps_assigned))
    for i in range(n_obj):
        if i not in maps_assigned:
            maps_left.append(i)
    # print("Maps left: ", maps_left)
    assignments = []
    if temp.depth+1 == n_agents:
        start = len(maps_left)
    else:
        start = 1

    for n in np.arange(start,len(maps_left)-(n_agents-temp.depth-1)+1):
        assignments = assignments + list(itertools.combinations(maps_left, n))

    return assignments

def update_upper_for_clustering(node,upper):
    # print("Trying to update the UB")
    radius = []
    temp = node
    while(temp):
        # print("\nCluster: ", temp.depth)
        # print("\nAssignment: ", temp.tasks)
        # print("\nRadius: ", temp.radius)
        if temp.depth == 0:
            break
        radius = radius + [temp.radius]
        # print("\nRadius all: ", radius)
        temp = temp.parent
        # breakpoint()
    if upper > max(radius):
        print("***updating upper bound")
        upper = max(radius)
    return upper

def find_best_allocation_for_clustering(root,values,alloc,radius_list):
    # list to store path
    if root == None:
        return 
    if len(root.children) == 0:
        # print("Reached leaf node")
        path = {}
        cluster_radii = []
        values.append(root)
        path[values[0].depth-1] = values[0].tasks
        if values[0].radius == np.inf:
            cluster_radii += [0]
        else:
            cluster_radii += [values[0].radius]
        for i in np.arange(1,len(values)):
            path[values[i].depth-1] = values[i].tasks
            cluster_radii += [values[i].radius]
        alloc.append(path)
        radius_list.append(cluster_radii)
        # print("Path found: ", path)
        # print("All paths: ", alloc)
        # print("All radius: ", radius_list)
        # print("Values: ", values)
        values.pop()
        return
    # print(str(root.agent)+" has "+str(len(root.children))+" children!")
    values.append(root)
    # print("Values: ", values)
    for child in root.children:
        find_best_allocation_for_clustering(child,values,alloc,radius_list)
    values.pop()

def bounding_sphere_clustering(problem,n_agents):
    """
    We want to minimize the maximum radius of the spheres. Each sphere represents one cluster. The upper bound here is the
    maximum radius of the sphere

    We do not have any incumbent solution, so initial upper bound is inf

    We will construct the BB tree with each level of the tree corresponding to one cluster (no assignment to agents yet)
    """
    start_time = time.time()

    n_obj = len(problem.pdfs)
    problem.nA = 100

    pdf_list = problem.pdfs

    #No incumbent clusters, so initial upper bound is infinity
    #Upper bound here is the maximum radius of the bounding spheres
    #Starting with any random clustering is really going to help compared to having nothing
    upper = np.inf #random_clustering(problem,n_agents)
    print("Current UB: ", upper)
    # breakpoint()

    #Start the tree with the root node being [], cluster -1
    root = ClusteringNode([], upper, [], None)

    # Nodes that are alive or not pruned
    explore_node = [root]
    pruned_combinations = {}
    not_pruned_combinations = {}
    nodes_pruned = 0

    for i in range(n_agents):
        print("Looking at combinations for cluster: ", i)
        
        new_explore_node = []
        for curr_node in explore_node:
            alloc_comb = generate_alloc_nodes_for_clustering(curr_node,n_obj,n_agents)
            # print("Parent combination: ", curr_node.tasks)
            # print("Possible combinations for cluster: ", alloc_comb)
            # breakpoint()

            for a in alloc_comb:                
                node = ClusteringNode(a,upper,[],curr_node)
                # print("Combination for this cluster: ", a)
                
                prune = False

                if a in not_pruned_combinations.keys():
                    # print("Already saw this combinations, not pruning")
                    node.alive = True
                    node.radius = not_pruned_combinations[a]
                    # breakpoint()

                elif a in pruned_combinations.keys():
                    # print("******************Already pruned before**********************")
                    node.alive = False
                    prune = True
                    nodes_pruned += 1
                else:
                    # The radius will almost be equal to zero when only one information map is considered
                    _, radius = get_minimal_bounding_sphere([pdf_list[j] for j in a],problem.nA,problem.pix)
                    node.radius = radius
                    # print("Radius: ", node.radius)
                    if radius > upper:
                        # print("Pruning the node")
                        node.alive = False
                        prune = True
                        pruned_combinations[a] = radius
                        nodes_pruned += 1
                if node.depth == n_agents:
                    if(node.alive):
                        upper = update_upper_for_clustering(node,upper)
                        # not_pruned_combinations = {}
                if not prune:
                    # print("Not pruning this node")
                    not_pruned_combinations[a] = node.radius
                    curr_node.children.append(node)
                    new_explore_node.append(node)
        explore_node = new_explore_node
        # breakpoint()

    print("Finished generating the tree")

    values = []
    alloc = []
    indv_radius = []
    find_best_allocation_for_clustering(root,values,alloc,indv_radius)
    # print("All paths found: ", alloc)
    # print("Individual ergodicities: ", indv_radius)
    # print("Number of agents: ", n_agents)
    print("Number of nodes pruned: ", nodes_pruned)
    # breakpoint()

    #Among all the combinations found, pick the clustering with minmax(radius)
    max_radius = []
    for idx,r in enumerate(indv_radius):
        if len(alloc[idx]) < n_agents+1:
            max_radius.append(100)
        else:
            max_radius.append(max(r))

    # print("Max radii: ", max_radius)
    min_idx = np.argmin(max_radius)

    best_clustering = alloc[min_idx]

    print("The best clustering according to minmax metric: ", best_clustering)
    # breakpoint()
    # print("Final allocation: ", final_allocation)
    runtime = time.time() - start_time
    return best_clustering,runtime
	
if __name__ == "__main__":
    n_agents = 4
    n_scalar = 10
    cnt = 0
    run_times = {}
    best_allocs = {}
    indv_erg_best = {}

    # best_alloc_done = np.load("./BB_similarity_clustering_random_maps_best_alloc_4_agents.npy",allow_pickle=True)
    # best_alloc_done = best_alloc_done.ravel()[0]

    for file in os.listdir("build_prob/random_maps/"):
        pbm_file = "build_prob/random_maps/"+file

        problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)

        if len(problem.pdfs) < n_agents:
            print("Less than 4")
            continue

        start_pos = np.load("./start_positions/start_pos_ang_random_4_agents.npy",allow_pickle=True)
        problem.s0 = start_pos.item().get(file)

        clusters_dict, runtime_clustering = bounding_sphere_clustering(problem,n_agents)
        print("The clusters are: ", clusters_dict)
        
        clusters = []
        for i in range(n_agents):
            clusters.append(np.asarray(clusters_dict[i]))
        print("Clusters: ", clusters)
        # breakpoint()
        
        best_alloc_OG, indv_erg_OG, start_pos_OG, runtime = branch_and_bound_main(problem,clusters,n_agents)
        
        print("File: ", file)
        print("\nBest allocation is: ", best_alloc_OG)
        print("\nBest Individual ergodicity: ", indv_erg_OG)
        
        run_times[file] = runtime
        best_allocs[file] = best_alloc_OG
        indv_erg_best[file] = indv_erg_OG

        # breakpoint()

        np.save("BB_bounding_sphere_clustering_random_maps_runtime_4_agents_remaining.npy", run_times)
        np.save("BB_bounding_sphere_clustering_random_maps_best_alloc_4_agents_remaining.npy", best_allocs)
        np.save("BB_bounding_sphere_clustering_random_maps_indv_erg_4_agents_remaining.npy", indv_erg_best)




