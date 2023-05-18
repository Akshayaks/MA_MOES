import matplotlib.pyplot as plt
import numpy as np

import common
import scalarize
from ergodic_coverage_hetero import ErgCover
import ergodic_metric_hetero
import random

import time
from utils import *
from explicit_allocation import scalarize_minmax
from miniball import miniball
import yaml

np.random.seed(101)

global agent_profile

""""
First find an incumbent solution using greedy approach to get an "upperd_bound". The incumbent solution
will be the max(indv_ergodicities) for the greedy allocation combination

Initialize a "dominant" set to []. This will house the definitely bad assignments (not entire allocation)

The root node is [], i.e. no assignment has been done so far
Then each level corresponds to the assignment of each agent to the maps

For each node, if the the max(erg) so far is > upper bound then discard that node

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

def greedy_alloc(problem, n_agents, agent_types, node = None, sensor_footprint = 15):
    #Allocate based on information in a window centered around the agent

    n_obj = len(problem.pdfs)
    print("Len objec: ", n_obj)
    agents_allotted = []
    maps_allotted = []
    if node:
        current_node = node
        while current_node.parent:
            agents_allotted.append(current_node.agent)
            maps_allotted = maps_allotted + list(current_node.tasks)
            current_node = current_node.parent

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

        for p in range(n_obj):
            if p in maps_allotted:
                continue
            agent_scores[i][p] = np.sum(problem.pdfs[p][y_min:y_max,x_min:x_max])

    allocation = {new_list: [] for new_list in range(n_agents)}

    agents_assigned = np.zeros(n_agents)
    print("agent scores: ", agent_scores)

    for i in range(n_obj):
        if i in maps_allotted:
            continue
        k = 0
        print("i: ", i)

        # Sort the amount of information on a map for different agents in descending order

        map_scores = [a[i] for a in agent_scores]
        info = sorted(map_scores, reverse=True) 
        found = False
        while not found:
            idx = map_scores.index(info[k]) # Free agent with max info on this map
            if (agents_assigned[idx] > 0 and np.count_nonzero(agents_assigned == 0) == n_obj - sum(agents_assigned)) or idx in agents_allotted:
                k += 1
            else:
                allocation[idx] = allocation[idx] + [i]
                found = True
                agents_assigned[idx] += 1
        
    print("Incumbent allocation: ", allocation)
    return allocation

def generate_alloc_nodes(curr_node,n_obj,n_agents):
    maps_assigned = []
    temp = curr_node
    while(curr_node):
        maps_assigned = maps_assigned + list(curr_node.tasks)
        curr_node = curr_node.parent
    print("Maps assigned: ", maps_assigned)
    maps_left = []
    for i in range(n_obj):
        if i not in maps_assigned:
            maps_left.append(i)
    print("Maps left: ", maps_left)
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
		result.append(path.copy())
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
		indv_erg = indv_erg + temp.indv_erg
		temp = temp.parent
	if upper > max(indv_erg):
		print("***updating upper bound***")
		upper = max(indv_erg)
	return upper

def find_best_allocation(root,values,alloc,indv_erg):
	# list to store path
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
		alloc.append(path)
		indv_erg.append(max_erg)
		values.pop()
		return
	values.append(root)
	for child in root.children:
		find_best_allocation(child,values,alloc,indv_erg)
	values.pop()

def get_subsets(s):
    subsets = []
    for i in range(1,len(s)):
        subsets = subsets + list(itertools.combinations(s, i))
    return subsets

# Do we need to normalize the pdf_FC we get by using the center of the minimal bounding sphere?
def get_minimal_bounding_sphere(pdf_list,nA,pix):
    FC = []
    for pdf in pdf_list:
        EC = ergodic_metric_hetero.ErgCalc(pdf.flatten(),1,nA,10,pix)
        FC.append(EC.phik*np.sqrt(EC.lamk))
    res = miniball(np.asarray(FC,dtype=np.double))
    pdf_FC = res["center"]
    pdf_FC = np.divide(res["center"],np.sqrt(EC.lamk))
    minmax = res["radius"]
    # print(res['radius'])
    # breakpoint()

    return pdf_FC, minmax

def branch_and_bound(pbm_file, n_agents, n_scalar, start_pos, agent_categories, random_start=False, scalarize=False, Bounding_sphere=False):

    start_time = time.time()
    pbm_file_complete = "./build_prob/random_maps/" + pbm_file
    problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)

    n_obj = len(problem.pdfs)
    problem.nA = 100

    if n_obj < n_agents:
        print("No < Na")
        return [],0,0,0

    #Generate random starting positions or read from file
    if random_start:
        pos = np.random.uniform(0,1,2*n_agents)
        theta = np.random.uniform(0,2*np.pi,n_agents)
        problem.s0 = []
        k = 0
        for i in range(n_agents):
            problem.s0.append(pos[k])
            problem.s0.append(pos[k+1])
            problem.s0.append(theta[i])
            k += 2
        problem.s0 = np.array(problem.s0)
    else:
        problem.s0 = start_pos.item().get(pbm_file)
    
    agent_types = agent_categories.item().get(pbm_file)
    # print("Agent categories: ", agent_categories)
    # print("Agent types: ", agent_types)
    # breakpoint()

    pdf_list = problem.pdfs

    print("Agent start positions allotted and agent type identified!")

    #Generate incumbent solution using Greedy approach
    incumbent = greedy_alloc(problem,n_agents,agent_types)
    incumbent_erg = np.zeros(n_obj)

    nodes_pruned = 0
    nodes_explored = 0

    pdf_FC = np.zeros((100,1))
    pdf = np.zeros((100,100))

    #Find the upper bound for the incumbent solution
    for k,v in incumbent.items():
        if len(v) > 1:
            if scalarize:
                pdf = scalarize_minmax([pdf_list[a] for a in v],problem.s0[k*3:k*3+3],problem.nA)
            if Bounding_sphere:
                print("Computing the center of the minimal bounding sphere")
                pdf_FC, _ = get_minimal_bounding_sphere([pdf_list[a] for a in v],problem.nA,problem.pix)
            else:
                print("Computing weighted average scalarized map")
                pdf = np.zeros((100,100))
                for a in v:
                    pdf += (1/len(v))*pdf_list[a]
        else:
            print("Single information map")
            pdf = pdf_list[v[0]]

        # if not Bounding_sphere:
        pdf = np.asarray(pdf.flatten())
        
        # Just run ergodicity optimization for fixed iterations to get ergodic trajectory
        control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], agent_types[k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)
        print("Erg: ", erg[-1])
        
        # Calculate individual ergodicities using the gotten trajectory
        for p in v:
            pdf_indv = np.asarray(pdf_list[p].flatten())
            EC = ergodic_metric_hetero.ErgCalc(pdf_indv,1,agent_types[k],problem.nA,n_scalar,problem.pix)
            incumbent_erg[p] = EC.fourier_ergodic_loss(control,problem.s0[3*k:3+3*k])

    upper = max(incumbent_erg)
    print("Incumbent allocation: ", incumbent)
    print("Incumber Ergodicities: ", incumbent_erg)
    print("Initial Upper: ", upper)
    # return {},0,0,0
    # breakpoint()

    #Start the tree with the root node being [], blank assignment
    root = Node(None, [], [], np.inf, np.inf, [], None)
    print("Added root node")

    # Nodes that are alive or not pruned
    explore_node = [root]
    single_map_agent = np.zeros((n_agents,n_obj))
    agent_alloc_pruned = [[] for _ in range(n_agents)]

    for i in range(n_agents):
        print("Looking at assignments for agent: ", i)
        agent_map_erg = {}
        
        new_explore_node = []
        for curr_node in explore_node:
            alloc_comb = generate_alloc_nodes(curr_node,n_obj,n_agents)
            print("Alloc_comb: ", alloc_comb)

            for a in alloc_comb:
                print("Looking at next possible assignment for agent: ", i)
                
                node = Node(i, a, [], np.inf, np.inf, [], curr_node)
                print("Alloc assigned: ", a)
                
                prune = False
                bad_alloc = False

                if a not in agent_map_erg.keys():
                    subsets = get_subsets(list(a))
                    for s in subsets:
                        if s in agent_alloc_pruned[i]:
                            print("\nAlloc contains subset of bad information map!")
                            agent_alloc_pruned[i].append(a)
                            node.alive = False
                            prune = True
                            nodes_pruned += 1 
                            bad_alloc = True
                            break
                    if bad_alloc:
                        continue
                    for ai in a:            
                        if single_map_agent[i][ai] == -1: #on indvidual map itself it does bad, it will definitely do bad on a set of maps involving this one
                            print("\nAlloc contains bad information map!")
                            agent_alloc_pruned[i].append(a)
                
                            node.alive = False
                            prune = True
                            nodes_pruned += 1 
                            bad_alloc = True
                            break
                    if bad_alloc:
                        continue
                    agent_map_erg[a] = []
                    if len(a) > 1:
                        if scalarize:
                            pdf = scalarize_minmax([pdf_list[j] for j in a],problem.s0[i*3:i*3+3],problem.nA)
                        elif Bounding_sphere:
                            pdf_FC, _ = get_minimal_bounding_sphere([pdf_list[ai] for ai in a],problem.nA,problem.pix)
                        else:
                            pdf = np.zeros((100,100))
                            for j in a:
                                pdf += (1/len(a))*pdf_list[j]
                    else:
                        print("Single information map")
                        pdf = pdf_list[a[0]]

                    pdf = np.asarray(pdf.flatten())

                    #Just run ergodicity optimization for fixed iterations and see which agent achieves best ergodicity in that time
                    if Bounding_sphere and len(a) > 1:
                        control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], agent_types[i], n_scalar, problem.pix, 1000, False, None, grad_criterion=True,direct_FC=pdf_FC)
                    else:
                        control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*i:3+3*i], agent_types[i], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)

                    #Can have a heuristic to show the order in which to evaluate the indv ergodicities
                    for p in a:
                        pdf_indv = np.asarray(pdf_list[p].flatten())
                        EC = ergodic_metric_hetero.ErgCalc(pdf_indv,1,agent_types[i],problem.nA,n_scalar,problem.pix)
                        erg = EC.fourier_ergodic_loss(control,problem.s0[3*i:3+3*i])
                        agent_map_erg[a].append(erg)
                        print("ERg: ", erg)
            
                        if erg > upper:
                            if len(a) == 1:
                                single_map_agent[i][a[0]] = -1
                            node.alive = False
                            prune = True
                            print("Don't explore further")
                            nodes_pruned += 1
                            agent_alloc_pruned[i].append(a) 
                            break
                        node.indv_erg.append(erg)
                else:
                    print("\nAlready saw this allocation!")
        
                    for e in agent_map_erg[a]:
                        if e > upper:
                            node.alive = False
                            prune = True
                            print("Don't explore further")
                            nodes_pruned += 1 
                            break
                        node.indv_erg.append(e)

                if node.depth == n_agents:
                    nodes_explored += 1
                    if(node.alive):
                        print("\nTrying to update the upper bound!")            
                        upper = update_upper(node,upper)
                if node.depth == n_agents:
                    nodes_explored += 1
                    if(node.alive):
                        print("\n******Trying to update the upper bound!")            
                        upper = update_upper(node,upper)
                if not prune:
                    print("Not pruning this node")
                    curr_node.children.append(node)
                    new_explore_node.append(node)

        explore_node = new_explore_node

    print("Number of nodes pruned: ", nodes_pruned)
    print("Number of nodes explored: ", nodes_explored)
    alloc_comb = generate_allocations(n_obj,n_agents)
    print("Total number of leaf nodes: ", len(alloc_comb))
    per_leaf_prunes = (len(alloc_comb) - nodes_explored)/len(alloc_comb)
    print("percentage of leaf nodes pruned: ", per_leaf_prunes)

    values = []
    alloc = []
    indv_erg = []
    find_best_allocation(root,values,alloc,indv_erg)
    print("All paths found: ", alloc)
    print("Individual ergodicities: ", indv_erg)
    print("Number of agents: ", n_agents)

    #Among all the allocations found, pick the one with min max erg
    max_erg = []
    for idx,e in enumerate(indv_erg):
        if len(alloc[idx]) < n_agents+1:
            max_erg.append(100)
        else:
            max_erg.append(max(e))

    print("Max ergodicities: ", max_erg)
    min_idx = np.argmin(max_erg)

    best_alloc = alloc[min_idx]

    print("The best allocation according to minmax metric: ", best_alloc)
    runtime = time.time() - start_time
    return best_alloc,runtime,per_leaf_prunes,indv_erg[min_idx]

def find_traj(file,alloc,problem,start_pos,agent_types,agent_profile):
    trajectories = []
    problem.s0 = start_pos.item().get(file)

    pdf_list = problem.pdfs

    for k,v in alloc.items():
        print(v)
        if k == None:
            continue
        if len(v) > 1:
            if scalarize:
                pdf = scalarize_minmax([pdf_list[a] for a in v],problem.s0[k*3:k*3+3],problem.nA)
            else:
                pdf = np.zeros((100,100))
                for a in v:
                    pdf += (1/len(v))*pdf_list[a]
        else:
            
            pdf = pdf_list[v[0]]

        pdf = np.asarray(pdf.flatten())
        
        # Just run ergodicity optimization for fixed iterations to get ergodic trajectory 
        control, _, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], agent_types[k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)

        speed = agent_profile["agent_type_speeds"][str(agent_types[k])]
        trajectories.append(ergodic_metric_hetero.GetTrajXY(control,problem.s0[3*k:3+3*k],speed))
    
    return trajectories

def collision_check(og_trajectories,alloc,problem,recheck=False):
    if not recheck:
        trajectories = []
        for i in range(len(og_trajectories)):
            trajectories.append(np.array(og_trajectories[i][1]))
        
        for i in range(len(trajectories)):
            for j in range(len(trajectories[i])):
                trajectories[i][j][0] = round(trajectories[i][j][0], 1)
                trajectories[i][j][1] = round(trajectories[i][j][1], 1)
    else:
        trajectories = og_trajectories
    
    priorities = []
    count = 0
    
    for i in range(len(trajectories)):
        for j in range(i+1,len(trajectories)):
            c = 0
            for k in range(1,len(trajectories[0])):
                c += 1
                
                if trajectories[i][k][0] == trajectories[j][k][0] and trajectories[i][k][1] == trajectories[j][k][1]:
                    count += 1
                    scorei = 0
                    scorej = 0
                    for map in alloc[i]:
                        scorei += problem.pdfs[map][round(trajectories[i][k][1])][round(trajectories[i][k][0])]
                    for map in alloc[j]:
                        scorej += problem.pdfs[map][round(trajectories[j][k][1])][round(trajectories[j][k][0])]
                    
                    if scorei >= scorej:
                        priorities.append((j,i,k)) #Make agent i wait
                        if k != 0:
                            l = k-1
                            while(trajectories[i][l][0] == trajectories[i][k][0] and trajectories[i][l][1] == trajectories[i][k][1]):
                                l -= 1
                            for m in range(l,k+1):
                                trajectories[i][m][0] = trajectories[i][l][0]
                                trajectories[i][m][1] = trajectories[i][l][1]
                    else:
                        priorities.append((i,j,k))
                        if k != 0:
                            l = k-1
                            while(trajectories[j][l][0] == trajectories[j][k][0] and trajectories[j][l][1] == trajectories[j][k][1]):
                                print(l,trajectories[j][l])
                                l -= 1
                            for m in range(l,k+1):
                                trajectories[j][m][0] = trajectories[j][l][0]
                                trajectories[j][m][1] = trajectories[j][l][1]
    print("Total number of collisions in the trajectory: ", count)
    print("Priorities gathered: ", priorities)
    return trajectories

## The way we display the collisions has to be corrected ##
def show_collisions(og_trajectories,problem,alloc):
    collision_points = []
    trajectories = []
    for i in range(len(og_trajectories)):
        trajectories.append(np.array(og_trajectories[i][1]))
    for i in range(len(trajectories)):
        for j in range(len(trajectories[i])):
            trajectories[i][j][0] = round(trajectories[i][j][0], 1)
            trajectories[i][j][1] = round(trajectories[i][j][1], 1)
    count = 0
    for i in range(len(trajectories)):
        for j in range(len(trajectories)):
            if j == i:
                continue
            c = 0
            for k in range(1,len(trajectories[0])):
                c += 1
                if trajectories[i][k][0] == trajectories[j][k][0] and trajectories[i][k][1] == trajectories[j][k][1]:
                    collision_points.append((k,og_trajectories[i][1][k][0],og_trajectories[i][1][k][1]))
                    count += 1
    tj = []
    for k in range(len(og_trajectories)):
        tj.append(np.array(og_trajectories[k][1]))

    print("Total number of collisions in the trajectory: ", count)
    display_map(problem,problem.s0,alloc,tj=tj,collision_points=collision_points)

    return        

if __name__ == "__main__":
    n_agents = 4
    n_scalar = 10
    run_times = {}
    best_allocs = {}
    per_leaf_prunes_list = {}
    indv_erg_best = {}

    ## Assign agent types randomly for each test case and each agent ##
    # Maximum allowed speed: 1,2 and 3 units per timestep ##
    
    # n_agent_types = 2 
    # agent_types = {}
    # for file in os.listdir("./build_prob/random_maps/"):
    #     types = []
    #     for i in range(n_agents):
    #         types.append(random.randrange(n_agent_types))
    #     agent_types[file] = types
    # np.save("Agent_types_4_agents.npy",agent_types)
    
    ########################################

    with open("agent_profile.yaml", "r") as yamlfile:
        agent_profile = yaml.load(yamlfile, Loader=yaml.FullLoader)
    
    # print("Read data: ", agent_profile)
    # breakpoint()

    start_pos = np.load("./start_positions/start_pos_ang_random_4_agents.npy",allow_pickle=True)
    agent_type = np.load("./Agent_types_4_agents.npy",allow_pickle=True)

    for file in os.listdir("build_prob/random_maps/"):
        pbm_file = "build_prob/random_maps/"+file

        problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)

        print("File: ", file)

        if len(problem.pdfs) < n_agents or len(problem.pdfs) > 4:
            continue

        # display_map_simple(problem,start_pos.item().get(file))

        final_allocation, runtime, per_leaf_prunes, indv_erg = branch_and_bound(file,n_agents,n_scalar,start_pos,agent_type,random_start=False, scalarize=False, Bounding_sphere=False)
        print("file: ", file)
        print("Agent type: ", agent_type.item().get(file))
        print("Final allocation: ", final_allocation)
        print("Runtime: ", runtime)
        print("per pruned: ", per_leaf_prunes)

        # breakpoint()

        run_times[pbm_file] = runtime
        best_allocs[pbm_file] = final_allocation
        per_leaf_prunes_list[pbm_file] = per_leaf_prunes
        indv_erg_best[pbm_file] = indv_erg

        trajectories = find_traj(file,final_allocation,problem,start_pos,agent_type.item().get(file),agent_profile)

        # print("Trajectories: ", trajectories)
        tj = []
        for i in range(len(trajectories)):
            tj.append(trajectories[i][1])

        display_map(problem,start_pos.item().get(file),final_allocation,tj=tj,title=str(agent_type.item().get(file))+file)

        breakpoint()
        # break

        # feasible_trajectories = collision_check(trajectories,final_allocation,problem)

        # print("Checking if the feasible trajectories is indeed collision free after modification")

        # new_traj = collision_check(feasible_trajectories,final_allocation,problem,recheck=True)

        # np.save("BB_opt_random_maps_runtime_4_agents_sphere.npy", run_times)
        # np.save("BB_opt_best_alloc_random_maps_4_agents_sphere.npy",best_allocs)
        # np.save("BB_opt_per_leaf_pruned_random_maps_4_agents_sphere.npy",per_leaf_prunes_list)
        # np.save("BB_opt_indv_erg_random_maps_4_agents_sphere.npy", indv_erg_best)


