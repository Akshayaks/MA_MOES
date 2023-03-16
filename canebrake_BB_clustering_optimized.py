import numpy as onp
import sys
import os

from jax import vmap, jit, grad
import jax.numpy as np
from jax.lax import scan
from functools import partial
import pdb
from sklearn.preprocessing import normalize
import common
import scalarize
from ergodic_coverage import ErgCover

import scipy.stats
import ergodic_metric
from utils import *
from explicit_allocation import *
import math
import time
import json

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from kneed import KneeLocator

np.random.seed(100)

"""
Branch and bound using clusters of maps based on the norm of the difference in their fourier 
coefficients. The agents are then allotted to one cluster instead of being allotted to a set
of maps. Every level of the tree will correspond to the assignment of an agent to one cluster.
Thus the tree will have a depth equal to the number of agents.
"""

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

def greedy_alloc(problem, clusters, n_agents, node = None):
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

    print("\nAgents already allotted: ", agents_allotted)
    print("\nMaps already assigned: ", clusters_allotted)
    if len(agents_allotted) == n_agents:
        print("all agents are allotted")
        return -1
    print("\nThere are some agents left to be allotted")
    print("\nCurrent problem start positions: ", problem.s0)
    # pdb.set_trace()

    sensor_footprint = 15
    agent_locs = []
    for i in range(n_agents):
        if i in agents_allotted:
            agent_locs.append((-1,-1))
        else:
            agent_locs.append((round(problem.s0[0+i*3]*100),round(problem.s0[1+i*3]*100)))
    print("Agent locations: ", agent_locs)

    agent_scores = np.zeros((n_agents,len(clusters)))

    #Calculate how much information each agent can get when allocatted to each map

    for i in range(n_agents):
        if i in agents_allotted:
            continue

        x_min = max(agent_locs[i][0]-sensor_footprint,0)
        x_max = min(agent_locs[i][0]+sensor_footprint,100)

        y_min = max(agent_locs[i][1]-sensor_footprint,0)
        y_max = min(agent_locs[i][1]+sensor_footprint,100)
        # print("agent: ", i)
        # print(xr,yr)

        for p in range(len(clusters)):
            if p in clusters_allotted:
                continue
            print("cluster being considered: ", p)
            for mapi in clusters[p]:
                # pdb.set_trace()
                agent_scores[i][p] += np.sum(problem.pdfs[mapi][y_min:y_max,x_min:x_max])
    print("Agent scores: ", agent_scores)
    # pdb.set_trace()

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
        # print("Allocation initialized")
        for i in range(len(clusters)):
            if i in clusters_allotted:
                continue
            k = 0
            erg = sorted([a[i] for a in agent_scores], reverse=True)
            # print("agent_scores: ",erg)
            found = False
            while not found:
                idx = [a[i] for a in agent_scores].index(erg[k])
                # print("idx: ", idx)
                #Find the bug in this line
                # print("agents_allotted: ", agents_allotted)
                # print("agent assigned[idx]: ", agents_assigned[idx])
                # print("n_obj-i-n_agents: ", n_obj-i-n_agents)
                zero_map_agents = list(agents_assigned).count(0)
                if (agents_assigned[idx] > 0 and len(clusters)-i == zero_map_agents) or idx in agents_allotted:
                    k += 1
                else:
                    allocation[idx] = allocation[idx] + [i]
                    found = True
                    agents_assigned[idx] += 1
                    # print("allocation so far: ", allocation)
    print("The final allocations are as follows: ", allocation)
    # pdb.set_trace()
    return allocation

def get_subsets(s):
    subsets = []
    for i in range(1,len(s)):
        subsets = subsets + list(itertools.combinations(s, i))
    return subsets

def branch_and_bound(problem, clusters, n_agents, start=[-1]):

    start_time = time.time()
    n_scalar = 10
    n_obj = len(problem.pdfs)
    # print("number of obj: ", n_obj)

    pdf_list = problem.pdfs
    problem.nA = 100

    print("Read start position as: ",problem.s0)

    #Generate incumbent solution using Greedy approach
    incumbent = greedy_alloc(problem,clusters,n_agents)
    print("\nGot incumbent cluster allocation!")

    incumbent_erg = np.zeros(n_obj)

    # final_allocation = {}
    nodes_pruned = 0

    #Find the upper bound for the incumbent solution
    for k,v in incumbent.items():
        # print("\nagent: ", k)
        # print("\ncluster: ", v)
        # print("\nmaps: ", clusters[v[0]])

        pdf = np.zeros((100,100))
        n_maps = 0
        for a in v:
            for mapi in clusters[a]:
                pdf += pdf_list[mapi]
                n_maps += 1
        pdf = (1/n_maps)*pdf

        # print("v: ", v)
        # if len(v) > 1:
        # 	pdf = np.zeros((100,100))
        # 	length = len(pdf_list)
        # 	for a in v:
        # 		pdf += (1/length)*pdf_list[a]
        # 	# scalarize_minmax([pdf_list[a] for a in v],problem.s0[k*3:k*3+3],problem.nA)
        # else:
        # 	pdf = pdf_list[v[0]]

        pdf = np.asarray(pdf.flatten())
        
        #Just run ergodicity optimization for fixed iterations and see which agent achieves best ergodicity in that time
        control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)
        print("\nmap erg: ", erg[-1])
        
        for p in v:
            for mapi in clusters[p]:
                print("\nmapi: ", mapi)
                pdf_indv = np.asarray(pdf_list[mapi].flatten())
                EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
                incumbent_erg[mapi] = EC.fourier_ergodic_loss(control,problem.s0[3*k:3+3*k])
                print("\nincum erg: ", incumbent_erg[mapi])

    upper = max(incumbent_erg)
    print("Incumbent allocation: ", incumbent)
    print("Incumber Ergodicities: ", incumbent_erg)
    print("Initial Upper: ", upper)
    # pdb.set_trace()

    nodes_explored = 0

    #Start the tree with the root node being [], blank assignment
    root = Node(None, [], [], [], np.inf, np.inf, [], None)
    # print("Added root node")

    # maps_assigned = np.zeros(n_obj)
    explore_node = [root]
    agent_alloc_pruned = [[] for _ in range(n_agents)]

    for i in range(n_agents):
        # print("Looking at assignments for agent: ", i)
        agent_cluster_erg = {}

        new_explore_node = []
        for curr_node in explore_node:
            alloc_cluster = generate_alloc_nodes(root,curr_node,n_agents,clusters)
            # print("Alloc_cluster: ", alloc_cluster)
            # pdb.set_trace()
            for a in alloc_cluster:
                print("Looking at next assignment for node: ", i)
                print("Parent assignment: ", curr_node.cluster)
                print("Assignment cluster: ", a)
                am = []
                for ai in a:
                    am = am + clusters[ai]
                print("Assignment maps: ", am)
                print("Nodes pruned so far: ", nodes_pruned)
                # pdb.set_trace()
                node = Node(i, am, a, [], np.inf, np.inf, [], curr_node)
                prune = False
                bad_alloc = False

                if a not in agent_cluster_erg.keys():
                    subsets = get_subsets(list(a))
                    for s in subsets:
                        if s in agent_alloc_pruned[i]:
                            print("\n**************************Alloc contains subset of bad information map!")
                            agent_alloc_pruned[i].append(a)
                            pdb.set_trace()
                            node.alive = False
                            prune = True
                            nodes_pruned += 1 
                            bad_alloc = True
                            break
                    if bad_alloc:
                        continue
                    agent_cluster_erg[a] = []

                    pdf = np.zeros((100,100))
                    n_maps = len(am)
                    for ai in a:
                        for mapi in clusters[ai]:
                            pdf += (1/n_maps)*pdf_list[mapi]
                    
                    pdf = np.asarray(pdf.flatten())

                    #Just run ergodicity optimization for fixed iterations and see which agent achieves best ergodicity in that time
                    control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*i:3+3*i], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)
                    print("\nmap erg: ", erg[-1])

                    for p in a:
                        for mapi in clusters[p]:
                            print("\n map: ", mapi)
                            pdf_indv = np.asarray(pdf_list[mapi].flatten())
                            EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
                            erg = EC.fourier_ergodic_loss(control,problem.s0[3*i:3+3*i])
                            agent_cluster_erg[a].append(erg)
                            print("\nerg: ", erg)
                            # pdb.set_trace()
                            if erg > upper:
                                node.alive = False
                                prune = True
                                print("Don't explore further")
                                # pdb.set_trace()
                                nodes_pruned += 1 
                                agent_alloc_pruned[i].append(a)
                                break
                            node.indv_erg.append(erg)
                        if prune:
                            break
                else:
                    print("\nAlready saw this allocation!")
                    # pdb.set_trace()
                    for e in agent_cluster_erg[a]:
                        if e > upper:
                            node.alive = False
                            prune = True
                            print("Don't explore further")
                            nodes_pruned += 1 
                            break
                        node.indv_erg.append(e)
                    # curr_node.children.append(node)
                if node.depth == n_agents:
                    nodes_explored += 1
                    if(node.alive):
                        print("\n******Trying to update the upper bound!")
                        print("\nparent agent: ", node.parent.agent)
                        print("\ncurrent node agent: ", curr_node.agent)
                        # pdb.set_trace()
                        upper = update_upper(node,upper)
                if not prune:
                    print("Not pruning this node")
                    node.cluster = a
                    node.tasks = am
                    curr_node.children.append(node)
                    print("node.indv_erg: ", node.indv_erg)
                    # pdb.set_trace()
                    new_explore_node.append(node)
        explore_node = new_explore_node

        # if i == n_agents-1:
        # 	print("Final agent assignments have been checked!")
        # 	for e in explore_node:
        # 		print(e.tasks)
    # final_allocation = find_best_allocation(root)

    # print("Number of nodes pruned: ", nodes_pruned)
    # total_alloc = len(generate_allocations(n_obj,n_agents))
    # print("Total number of nodes: ", total_alloc*n_agents)
    # print("Percentage of nodes pruned: ", nodes_pruned/total_alloc)
    alloc_comb = generate_allocations(len(clusters),n_agents)
    print("Total number of leaf nodes: ", len(alloc_comb))
    per_leaf_prunes = (len(alloc_comb) - nodes_explored)/len(alloc_comb)
    return root, problem.s0, per_leaf_prunes

def branch_and_bound_main(pbm,clusters,n_agents,start_pos=[-1]):
	start_time = time.time()
	root, start_pos, per_leaf_pruned = branch_and_bound(pbm,clusters,n_agents,start_pos)

	values = []
	alloc = []
	indv_erg = []
	find_best_allocation(root,values,alloc,indv_erg)
	print("All paths found: ", alloc)
	print("Individual ergodicities: ", indv_erg)
	print("Number of agents: ", n_agents)
	print("Number of clusters: ", clusters)
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
	# final_pdfs = []
	# final_trajectories = []
	# for i in range(n_agents):
	# 	pdf = np.zeros((100,100))
	# 	for mi in best_alloc[i]:
	# 		print(mi)
	# 		pdf += (1/len(best_alloc[i]))*pbm.pdfs[mi]
	# 	final_pdfs.append(pdf)
	
	# print("Got final pdfs")
	
	# for i in range(n_agents):
	# 	pdfi = np.asarray(final_pdfs[i].flatten())
	# 	#Just run ergodicity optimization for fixed iterations and see which agent achieves best ergodicity in that time
	# 	control, _, _ = ErgCover(pdfi, 1, pbm.nA, pbm.s0[3*i:3+3*i], n_scalar, pbm.pix, 1000, False, None, grad_criterion=True)
	# 	_, tj = ergodic_metric.GetTrajXY(control,pbm.s0[3*i:3+3*i])
	# 	final_trajectories.append(tj)
	# print(final_trajectories)
	# print(len(pbm.pdfs))
	# pbm_final = pbm
	# pbm_final.pdfs = final_pdfs
	# pdb.set_trace()
	# display_map(pbm_final,pbm_final.s0,pbm_file=None,tj=final_trajectories,window=None,r=None,title=None)
	# traj = [final_trajectories[2],final_trajectories[0],final_trajectories[1],final_trajectories[2],final_trajectories[0]]
	# display_map(pbm,pbm.s0,pbm_file=None,tj=traj,window=None,r=None,title=None)
	# print("Final allocation: ", final_allocation)
	runtime = time.time() - start_time
	return best_alloc,indv_erg[min_idx],start_pos,runtime, per_leaf_pruned



def jensen_shannon_distance(p, q):
	"""
	method to compute the Jenson-Shannon Distance 
	between two probability distributions
	"""

	# convert the vectors into numpy arrays in case that they aren't
	p = p.flatten()
	q = q.flatten()
	d = 0.2

	print("Similarity using Jensen Shannon Distance:")
	all_dist = []

	for i in range(10):
		# calculate m
		m = (p + (p+d)) / 2

		# compute Jensen Shannon Divergence
		divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

		# compute the Jensen Shannon Distance
		distance = np.sqrt(divergence)
		print("distance: ", distance)
		d += 0.2
		all_dist.append(distance*100)

	return distance, all_dist

def ergodic_similarity(problem, n_scalar): #n_scalar -> num of fourier coefficients
	pdf1 = problem.pdfs[0].flatten()
	pdf2 = problem.pdfs[1].flatten()

	EC1 = ergodic_metric.ErgCalc(pdf1,1,problem.nA,n_scalar,problem.pix)

	d = 0.2
	all_dist = []

	for i in range(10):
		print(sum(pdf1))
		print(sum((pdf1+d)/sum(pdf1+d)))
		# pdb.set_trace()
		EC2 = ergodic_metric.ErgCalc((pdf1+d)/sum(pdf1+d),1,problem.nA,n_scalar,problem.pix)

		distance = np.sum(EC1.lamk*np.square(EC1.phik - EC2.phik))
		print("Distance: ", distance)
		d += 0.2
		all_dist.append(distance*1000)

	return distance, all_dist

def k_means_clustering(pbm,n_agents,n_scalar):
	
	pdfs = pbm.pdfs
	data = [pdf.flatten() for pdf in pdfs]
	data = []
	# n_scalar = 10
	for pdf in pdfs:
		EC = ergodic_metric.ErgCalc(pdf.flatten(),1,pbm.nA,n_scalar,pbm.pix)
		data.append(EC.phik)
	cost =[]
	for i in range(1, len(pdfs)+1):
		KM = KMeans(n_clusters = i, max_iter = 500)
		KM.fit(data)
		# calculates squared error
		# for the clustered points
		cost.append(KM.inertia_*100000)
		print("\nCost for this clustering: ", cost)
		#plot the cost against K values
	x = np.arange(1,len(pdfs)+1)
	kn = KneeLocator(x, cost, curve='convex', direction='decreasing')
	print("knee: ", kn.knee)

	##### For now we want the n_clusters >= n_agents ########
	if kn.knee:
		if kn.knee < n_agents:
			print("\n*************Forcing number of clusters to be equal to number of agents!!********\n")
			n_clusters = n_agents
		else:
			n_clusters = kn.knee
	else:
		n_clusters = n_agents

	# plt.plot(range(1, len(pdfs)+1), cost, color ='g', linewidth ='3')
	# plt.xlabel("Value of K")
	# plt.ylabel("Squared Error (Cost)")
	# plt.show() # clear the plot
	Kmean = KMeans(n_clusters=n_clusters)
	Kmean.fit(data)
	print("\nThe cluster labels are: ", Kmean.labels_)
	# pdb.set_trace()
	clusters = [[] for _ in range(n_clusters)]
	labels = Kmean.labels_
	for idx,l in enumerate(labels):
		clusters[l].append(idx)
	
	print("\nFinal clusters are: ", clusters)
	# pdb.set_trace()

	return clusters
	
if __name__ == "__main__":
    n_agents = 3
    n_scalar = 10
    cnt = 0

    file1 = open("similarity_based_BandB.txt","w")
    run_times = {}
    best_allocs = {}
    indv_erg_best = {}
    per_pruning = {}

    for file in os.listdir("build_prob/random_maps/"):
        # if cnt == 2:
        # 	break
        pbm_file = "build_prob/random_maps/"+file

        print("\nFile: ", file)
        # pdb.set_trace()

        if file != "random_map_55.pickle":
            continue

        problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)

        if len(problem.pdfs) < n_agents:
            continue

        start_pos = np.load("start_pos_ang_random_3_agents.npy",allow_pickle=True)
        problem.s0 = start_pos.item().get(file)

        if len(problem.pdfs) == n_agents:
            clusters = [[i] for i in range(n_agents)]
        else:
            clusters = k_means_clustering(problem,n_agents,n_scalar)
        print("The clusters are: ", clusters)
        # pdb.set_trace()

        best_alloc_OG, indv_erg_OG, start_pos_OG, runtime, per_leaf_pruned = branch_and_bound_main(problem,clusters,n_agents)
        # for i in range(n_agents):
        # 	if i in best_alloc_OG.keys():
        # 		best_alloc_OG[i] = clusters[best_alloc_OG[i][0]]
        
        print("\nBest allocation is: ", best_alloc_OG)
        print("\nBest Individual ergodicity: ", indv_erg_OG)
        
        run_times[file] = runtime
        best_allocs[file] = best_alloc_OG
        indv_erg_best[file] = indv_erg_OG
        per_pruning[file] = per_leaf_pruned

        np.save("BB_similarity_clustering_runtime_3_agents.npy", run_times)
        np.save("BB_similarity_clustering_best_alloc_3_agents.npy", best_allocs)
        np.save("BB_similarity_clustering_indv_erg_3_agents.npy", indv_erg_best)
        np.save("BB_similarity_clustering_per_pruned_3_agents.npy", per_pruning)

        file1.write(file)
        file1.write("\n")
        file1.write(json.dumps(best_alloc_OG))
        file1.write("\n")
        file1.write("clusters: ")
        for c in clusters:
            for ci in c:
                file1.write(str(ci))
                file1.write(", ")
            file1.write("; ")
        file1.write("\n")
        # cnt += 1
        # break
        # pdb.set_trace()

    file1.close()





