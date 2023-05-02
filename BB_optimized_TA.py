import matplotlib.pyplot as plt
import numpy as np

import common
from ergodic_coverage import ErgCover
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

    for i in range(n_obj):
        if i in maps_allotted:
            continue
        k = 0
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

def branch_and_bound(pbm_file, n_agents, n_scalar, start_pos, random_start=True, scalarize=False):

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

    # display_map(problem,problem.s0)

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

        pdf = np.asarray(pdf.flatten())
        
        # Just run ergodicity optimization for fixed iterations to get ergodic trajectory 
        control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)
        
        # Calculate individual ergodicities using the gotten trajectory
        for p in v:
            pdf_indv = np.asarray(pdf_list[p].flatten())
            EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
            incumbent_erg[p] = EC.fourier_ergodic_loss(control,problem.s0[3*k:3+3*k])

    upper = max(incumbent_erg)
    print("Incumbent allocation: ", incumbent)
    print("Incumber Ergodicities: ", incumbent_erg)
    print("Initial Upper: ", upper)

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
                        else:
                            pdf = np.zeros((100,100))
                            for j in a:
                                pdf += (1/len(a))*pdf_list[j]
                    else:
                        pdf = pdf_list[a[0]]

                    pdf = np.asarray(pdf.flatten())

                    #Just run ergodicity optimization for fixed iterations and see which agent achieves best ergodicity in that time
                    control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*i:3+3*i], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)

                    #Can have a heuristic to show the order in which to evaluate the indv ergodicities
                    for p in a:
                        pdf_indv = np.asarray(pdf_list[p].flatten())
                        EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
                        erg = EC.fourier_ergodic_loss(control,problem.s0[3*i:3+3*i])
                        agent_map_erg[a].append(erg)
                        print("erg: ", erg)
                       
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
                        print("\nTrying to update the upper bound!")
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

    k_best_allocs = find_k_best_allocs(alloc,indv_erg,n_agents)

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
    return best_alloc,runtime,per_leaf_prunes,indv_erg[min_idx],k_best_allocs

def find_k_best_allocs(alloc,indv_erg,n_agents):
    complete_allocs = []
    for idx,e in enumerate(indv_erg):
        if len(alloc[idx]) == n_agents+1:
            complete_allocs.append((max(e),alloc[idx]))
    complete_allocs.sort(key = lambda x: x[0])
    print("K best allocations: ", complete_allocs)
    return complete_allocs
        

def find_traj(file,alloc,problem,start_pos,scalarize=False):
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
        control, _, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)

        trajectories.append(ergodic_metric.GetTrajXY(control,problem.s0[3*k:3+3*k]))
    
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

## This function has to be corrected to display the collision points ##
def show_collisions(og_trajectories,problem,recheck=False):
    collision_points = []
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
    count = 0
    for i in range(len(trajectories)):
        for j in range(i+1,len(trajectories)):
            c = 0
            for k in range(1,len(trajectories[0])):
                c += 1
                if trajectories[i][k][0] == trajectories[j][k][0] and trajectories[i][k][1] == trajectories[j][k][1]:
                    collision_points.append((trajectories[i][k][0],trajectories[i][k][1]))
                    count += 1

    print("Total number of collisions in the trajectory: ", count)
    display_map(problem,problem.s0,collision_points=collision_points)

    return 

def compute_max_indv_erg(file,final_allocation,problem,start_pos):
    trajectories = find_traj(file,final_allocation,problem,start_pos)

    tj = []
    for i in range(len(trajectories)):
        tj.append(trajectories[i][1])

    display_map(problem,start_pos.item().get(file),final_allocation,tj=tj)

    # show_collisions(trajectories,problem)
     
    feasible_trajectories = collision_check(trajectories,final_allocation,problem)

    indv_erg = []

    problem.s0 = start_pos.item().get(file)

    pdf_list = problem.pdfs

    for k,v in final_allocation.items():
        for p in v:
            pdf_indv = np.asarray(pdf_list[p].flatten())
            EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
            erg = EC.fourier_ergodic_loss_traj(feasible_trajectories[k])
            indv_erg.append(erg)
    return max(indv_erg) 

if __name__ == "__main__":
    n_agents = 4
    n_scalar = 10
    run_times = {}
    best_allocs = {}
    per_leaf_prunes = {}
    indv_erg_best = {}

    start_pos = np.load("./start_positions/start_pos_ang_random_4_agents.npy",allow_pickle=True)
    for file in os.listdir("build_prob/random_maps/"):
        pbm_file = "build_prob/random_maps/"+file
        print("\nFile: ", pbm_file)
        problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)

        if len(problem.pdfs) != n_agents:
            continue

        final_allocation, runtime, per_leaf_prunes, indv_erg, k_best_allocs = branch_and_bound(file,n_agents,n_scalar,start_pos,random_start=False, scalarize=False)
        print("file: ", file)
        print("Final allocation: ", final_allocation)
        print("Runtime: ", runtime)
        print("per pruned: ", per_leaf_prunes)

        # run_times[pbm_file] = runtime
        # best_allocs[pbm_file] = final_allocation
        # per_leaf_prunes[pbm_file] = per_leaf_prunes
        # indv_erg_best[pbm_file] = indv_erg   

        current_alloc_idx = 0 
        alloc_minmax = {}
        found_best_alloc = False
        best_minmax_so_far = np.inf
        best_alloc_idx = -1

        while(not found_best_alloc):
            print("Allocation number: ", current_alloc_idx)
            print("Allocation considered: ", k_best_allocs[current_alloc_idx])
            alloc_minmax[current_alloc_idx] = compute_max_indv_erg(file,k_best_allocs[current_alloc_idx][1],problem, start_pos)
            if current_alloc_idx == 0:
                best_minmax_so_far = alloc_minmax[current_alloc_idx]
                best_alloc_idx = current_alloc_idx
                print("Computed cost of the best allocation")
            else:
                if alloc_minmax[current_alloc_idx] >= best_minmax_so_far:
                    print("Found the best allocation scheme with collision-free trajectories")
                    found_best_alloc = True
                else:
                    best_minmax_so_far = alloc_minmax[current_alloc_idx]
                    best_alloc_idx = current_alloc_idx
            current_alloc_idx += 1
        
        print("Final best allocation: ", k_best_allocs[best_alloc_idx])
        print("Final minmax: ", best_minmax_so_far)

        # np.save("BB_opt_random_maps_sparse_runtime_4_agents.npy", run_times)
        # np.save("BB_opt_best_alloc_random_maps_sparse_4_agents.npy",best_allocs)
        # np.save("BB_opt_per_leaf_pruned_random_maps_sparse_4_agents.npy",per_leaf_prunes)
        # np.save("BB_opt_indv_erg_random_maps_sparse_4_agents.npy", indv_erg_best)

