# import matplotlib.pyplot as plt
import numpy as np

import common
# from ergodic_coverage import ErgCover
import ergodic_metric
from miniball import miniball

import time
from utils import *
from explicit_allocation import *

np.random.seed(101)

""""
Try to do clustering using minimum bound spheres instead of K-means on features because, we care about minimizing the maximum
distance of elements in the cluster rather than sum of squared distances

"""

class Node:
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
    

def generate_alloc_nodes(curr_node,n_obj,n_agents):
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

def update_upper(node,upper):
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

def find_best_allocation(root,values,alloc,radius_list):
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
        find_best_allocation(child,values,alloc,radius_list)
    values.pop()

def get_subsets(s):
    subsets = []
    for i in range(1,len(s)):
        subsets = subsets + list(itertools.combinations(s, i))
    return subsets 

def random_clustering(problem,n_agents):
    #Divide the maps equally among the agents
    n_maps = int(len(problem.pdfs)/n_agents)
    radii = []
    l = 0
    for i in range(n_agents):
        pdf_list = []
        j = l
        while j < l+n_maps:
            pdf_list.append(problem.pdfs[j])
            j += 1
        l += n_maps
        if i == n_agents-1 and l < len(problem.pdfs):
            pdf_list += problem.pdfs[l:]
            # while l < len(problem.pdfs):
            #     pdf_list.append(problem.pdfs[l])
        # print("Len pdf_list: ", len(pdf_list))
        # breakpoint()
        _, r = get_minimal_bounding_sphere(pdf_list,problem.nA,problem.pix)
        radii.append(r)
        # print("radii: ", radii)
        # breakpoint()

    return max(radii)

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
    root = Node([], upper, [], None)

    # Nodes that are alive or not pruned
    explore_node = [root]
    pruned_combinations = {}
    not_pruned_combinations = {}
    nodes_pruned = 0

    for i in range(n_agents):
        print("Looking at combinations for cluster: ", i)
        
        new_explore_node = []
        for curr_node in explore_node:
            alloc_comb = generate_alloc_nodes(curr_node,n_obj,n_agents)
            # print("Parent combination: ", curr_node.tasks)
            # print("Possible combinations for cluster: ", alloc_comb)
            # breakpoint()

            for a in alloc_comb:                
                node = Node(a,upper,[],curr_node)
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
                        upper = update_upper(node,upper)
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
    find_best_allocation(root,values,alloc,indv_radius)
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
    # runtime = time.time() - start_time
    return best_clustering #,runtime,per_leaf_prunes,indv_radius[min_idx]



if __name__ == "__main__":
    n_agents = 4
    n_scalar = 10
    run_times = {}
    best_allocs = {}
    per_leaf_prunes = {}
    indv_erg_best = {}

    best_alloc_bb = np.load("./results_canebrake/BB_opt_Best_alloc_4_agents.npy",allow_pickle=True)
    best_alloc_bb = best_alloc_bb.ravel()[0]

    best_alloc_sim = np.load("./Results_npy_files/BB_similarity_clustering_random_maps_best_alloc_4_agents.npy",allow_pickle=True)
    best_alloc_sim = best_alloc_sim.ravel()[0]

    alloc_clustering = np.load("./Results_npy_files/Best_clustering_minimal_bounding_sphere_4_agents.npy",allow_pickle=True)
    alloc_clustering = alloc_clustering.ravel()[0]

    best_clustering = {}

    # start_pos = np.load("./start_pos_ang_random_4_agents.npy",allow_pickle=True)
    for file in os.listdir("build_prob/random_maps/"):
        
        pbm_file = "build_prob/random_maps/"+file
        # if file not in best_alloc_bb.keys() or file not in best_alloc_sim.keys():
        #     continue
        print("\nFile: ", file)
        
        problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)
        print("\nNumber of maps: ", len(problem.pdfs))

        if len(problem.pdfs) <= n_agents or len(problem.pdfs) > 7:
            continue

        clusters = bounding_sphere_clustering(problem,n_agents)

        print("Clusters from BB: ", best_alloc_bb[file])
        print("Clusters from minimal bounding spheres: ", clusters)
        print("Clusters from similarity clustering: ", best_alloc_sim[file])
        # breakpoint()

        best_clustering[file] = clusters

        np.save("Best_clustering_minimal_bounding_sphere.npy",best_clustering)


