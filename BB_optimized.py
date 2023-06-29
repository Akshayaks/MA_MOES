import matplotlib.pyplot as plt
import numpy as np
import common
import scalarize
from ergodic_coverage import ErgCover
import ergodic_metric
import time
from utils import *
from explicit_allocation import scalarize_minmax
from miniball import miniball

np.random.seed(101)

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
        
    # print("Incumbent allocation: ", allocation)
    return allocation

def generate_alloc_nodes(curr_node,n_obj,n_agents):
    maps_assigned = []
    temp = curr_node
    while(curr_node):
        maps_assigned = maps_assigned + list(curr_node.tasks)
        curr_node = curr_node.parent
    # print("Maps assigned: ", maps_assigned)
    maps_left = []
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
		# print("***updating upper bound***")
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
    # if len(s) == 1:
    #     return np.array(s)
    for i in range(1,len(s)):
        subsets = subsets + list(itertools.combinations(s, i))
    return subsets

def get_minimal_bounding_sphere(pdf_list,nA,pix):
    FC = []
    for pdf in pdf_list:
        EC = ergodic_metric.ErgCalc(pdf.flatten(),1,nA,10,pix)
        FC.append(EC.phik*np.sqrt(EC.lamk))

    res = miniball(np.asarray(FC,dtype=np.double))
    pdf_FC = res["center"]
    pdf_FC = np.divide(res["center"],np.sqrt(EC.lamk))
    minmax = res["radius"]
    return pdf_FC, minmax

def branch_and_bound(pbm_file, n_agents, n_scalar, start_pos, random_start=False, scalarize=False, Bounding_sphere=False):

    start_time = time.time()
    pbm_file_complete = "./build_prob/random_maps/" + pbm_file
    problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)

    n_obj = len(problem.pdfs)
    problem.nA = 100

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

    pdf_list = problem.pdfs

    #Generate incumbent solution using Greedy approach
    incumbent = greedy_alloc(problem,n_agents)
    incumbent_erg = np.zeros(n_obj)

    nodes_pruned = 0
    nodes_explored = 0
    pdf = np.zeros((100,100))

    #Find the upper bound for the incumbent solution
    for k,v in incumbent.items():
        if len(v) > 1:
            if scalarize:
                pdf = scalarize_minmax([pdf_list[a] for a in v],problem.s0[k*3:k*3+3],problem.nA)
            if Bounding_sphere:
                # print("Computing MBS")
                pdf_center, _ = get_minimal_bounding_sphere([pdf_list[a] for a in v],problem.nA,problem.pix)
            else:
                pdf = np.zeros((100,100))
                for a in v:
                    pdf += (1/len(v))*pdf_list[a]
        else:
            pdf = pdf_list[v[0]]

        # if not Bounding_sphere:
        pdf = np.asarray(pdf.flatten())
        
        # Just run ergodicity optimization for fixed iterations to get ergodic trajectory
        if Bounding_sphere and len(v) > 1:
           control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True,direct_FC=pdf_center)
        else: 
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
    # breakpoint()

    #Start the tree with the root node being [], blank assignment
    root = Node(None, [], [], np.inf, np.inf, [], None)

    # Nodes that are alive or not pruned
    explore_node = [root]
    single_map_agent = np.zeros((n_agents,n_obj))
    # single_map_agent = np.zeros((n_agents,n_obj))
    agent_alloc_pruned = [[] for _ in range(n_agents)]

    for i in range(n_agents):
        # print("Agent: ", i)
        agent_map_erg = {}
        new_explore_node = []
        for curr_node in explore_node:
            alloc_comb = generate_alloc_nodes(curr_node,n_obj,n_agents)

            for a in alloc_comb:
                # print("Alloc: ", a)
                node = Node(i, a, [], np.inf, np.inf, [], curr_node)                
                prune = False
                bad_alloc = False

                if a not in agent_map_erg.keys():
                    subsets = get_subsets(list(a))

                    for s in subsets:
                        if s in agent_alloc_pruned[i]:
                            # print("\nAlloc contains a bad subset of info maps")
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
                            # print("\nAlloc contains bad information map!")
                            agent_alloc_pruned[i].append(a)
                            node.alive = False
                            prune = True
                            nodes_pruned += 1 
                            bad_alloc = True
                            break
                    if bad_alloc:
                        continue
                    agent_map_erg[a] = []
                    
                    pdf = np.zeros((100,100))
                    if len(a) > 1:
                        if scalarize:
                            pdf = scalarize_minmax([pdf_list[j] for j in a],problem.s0[i*3:i*3+3],problem.nA)
                        elif Bounding_sphere:
                            # print("Finding MBS")
                            pdf_center, _ = get_minimal_bounding_sphere([pdf_list[j] for j in a],problem.nA,problem.pix)
                        else:
                            pdf = np.zeros((100,100))
                            for j in a:
                                pdf += (1/len(a))*pdf_list[j]
                    else:
                        pdf = pdf_list[a[0]]

                    pdf = np.asarray(pdf.flatten())

                    #Just run ergodicity optimization for fixed iterations and see which agent achieves best ergodicity in that time
                    if Bounding_sphere and len(a) > 1:
                        control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*i:3+3*i], n_scalar, problem.pix, 1000, False, None, grad_criterion=True,direct_FC=pdf_center)
                        # print("Erg on scal: ", erg[-1])
                    else:
                        control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*i:3+3*i], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)

                    #Can have a heuristic to show the order in which to evaluate the indv ergodicities
                    # print("Computing indv erg for: ", a)
                    for p in a:
                        pdf_indv = np.asarray(pdf_list[p].flatten())
                        EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
                        erg = EC.fourier_ergodic_loss(control,problem.s0[3*i:3+3*i])
                        agent_map_erg[a].append(erg)
                        # print(p,erg)
            
                        if erg > upper:
                            if len(a) == 1:
                                single_map_agent[i][a[0]] = -1
                            node.alive = False
                            prune = True
                            # print("Don't explore further")
                            nodes_pruned += 1
                            agent_alloc_pruned[i].append(a) 
                            break
                        node.indv_erg.append(erg)
                else:
                    # print("\nAlready saw this allocation!")
                    for e in agent_map_erg[a]:
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
                        # print("\nTrying to update the upper bound!")            
                        upper = update_upper(node,upper)
                if not prune:
                    # print("Not pruning this node")
                    curr_node.children.append(node)
                    new_explore_node.append(node)

        explore_node = new_explore_node
        # breakpoint()

    alloc_comb = generate_allocations(n_obj,n_agents)
    per_leaf_prunes = (len(alloc_comb) - nodes_explored)/len(alloc_comb)

    values = []
    alloc = []
    indv_erg = []
    find_best_allocation(root,values,alloc,indv_erg)
    # print("All paths found: ", alloc)
    # breakpoint()
    correct_erg = []
    correct_alloc = []
    for i in range(len(alloc)):
        if len(alloc[i]) == n_agents + 1:
            correct_erg.append(indv_erg[i])
            correct_alloc.append(alloc[i])
    # breakpoint()
    if len(correct_erg) == 0:
        best_alloc = incumbent
        runtime = time.time() - start_time
        print("Check this file")
        # breakpoint()
        return best_alloc,runtime,per_leaf_prunes,incumbent_erg
    else:
        min_idx = find_minmax(correct_erg)
        best_alloc = correct_alloc[min_idx]
        # print("The best allocation according to minmax metric: ", best_alloc)
        runtime = time.time() - start_time
                
        # breakpoint()
        return best_alloc,runtime,per_leaf_prunes,correct_erg[min_idx]

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

def find_traj(file,alloc,problem,start_pos,Bounding_sphere=True):
    trajectories = []
    problem.s0 = start_pos.item().get(file)

    pdf_list = problem.pdfs

    for k,v in alloc.items():
        if k == None:
            continue
        if len(v) > 1:
            if scalarize:
                pdf = scalarize_minmax([pdf_list[a] for a in v],problem.s0[k*3:k*3+3],problem.nA)
            if Bounding_sphere:
                pdf_center, _ = get_minimal_bounding_sphere([pdf_list[a] for a in v],problem.nA,problem.pix)
            else:
                pdf = np.zeros((100,100))
                for a in v:
                    pdf += (1/len(v))*pdf_list[a]
        else: 
            pdf = pdf_list[v[0]]

        pdf = np.asarray(pdf.flatten())
        
        # Just run ergodicity optimization for fixed iterations to get ergodic trajectory 
        if Bounding_sphere and len(v) > 1:
            control, _, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True,direct_FC=pdf_center)
        else:
            control, _, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)

        trajectories.append(ergodic_metric.GetTrajXY(control,problem.s0[3*k:3+3*k]))
    
    return trajectories         

if __name__ == "__main__":
    n_agents = 4
    n_scalar = 10
    run_times = {}
    best_allocs = {}
    per_leaf_prunes_list = {}
    indv_erg_best = {}
    best_traj = {}

    # done = np.load("./Final_exp/BB_MBS_scal_runtime.npy", allow_pickle=True)
    # done = done.ravel()[0]

    start_pos = np.load("./start_positions/start_pos_ang_random_4_agents.npy",allow_pickle=True)
    for file in os.listdir("build_prob/random_maps/"):
        pbm_file = "build_prob/random_maps/"+file
        print("\nFile: ", pbm_file)
        problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)

        if len(problem.pdfs) < 9 or len(problem.pdfs) > 10:
            print("Na < No")
            continue

        # if file in done.keys():
        #     continue

        # if file != "random_map_28.pickle":
        #     continue

        print("No: ", len(problem.pdfs))

        # display_map(problem,start_pos.item().get(file),{0:[0],1:[1],2:[2],3:[3]},pbm_file=file)

        final_allocation, runtime, per_leaf_prunes, indv_erg = branch_and_bound(file,n_agents,n_scalar,start_pos,random_start=False,scalarize=False,Bounding_sphere=True)
        print("file: ", file)
        print("Final allocation: ", final_allocation)
        print("Runtime: ", runtime)
        print("per pruned: ", per_leaf_prunes)
        print("indv_erg: ", indv_erg)
        print("minmax metric: ", max(indv_erg))
        print("*********************************************************************")

        run_times[file] = runtime
        best_allocs[file] = final_allocation
        per_leaf_prunes_list[file] = per_leaf_prunes
        indv_erg_best[file] = indv_erg

        trajectories = find_traj(file,final_allocation,problem,start_pos)

        # We got the trajectory of each agent, now get trajectories for each map based on allocation
        tj = []
        for i in range(len(problem.pdfs)):
            for j in range(n_agents):
                if i in final_allocation[j]:
                    tj.append(trajectories[j][1])
                    break

        best_traj[file] = tj

        display_map(problem,start_pos.item().get(file),final_allocation,pbm_file=file,tj=tj)

        np.save("Final_exp/BB_MBS_scal_runtime_remaining.npy", run_times)
        np.save("Final_exp/BB_MBS_scal_alloc_remaining.npy",best_allocs)
        np.save("Final_exp/BB_MBS_scal_pruned_remaining.npy",per_leaf_prunes_list)
        np.save("Final_exp/BB_MBS_scal_indv_erg_remaining.npy", indv_erg_best)
        np.save("Final_exp/BB_MBS_scal_traj_remaining.npy",best_traj)


