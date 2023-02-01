from explicit_allocation import *
import os
import pdb
import jax
import argparse
import common
from branch_and_bound import branch_and_bound 
'''
Iterate through all the example problems in test_cases and execute MOES on them and store the result
Currently doing it just for 2 agents
'''

def run_exp_bb(folder,start_pos_file,n_agents):
	# Check in how many cases fathoming a "promising" node actually led to an improvement in the upper bound. 
	# Maybe we have to change how we are using the score to decide if we want to fathom a node
	run_times = {}
	best_allocs = {}
	per_leaf_prunes = {}
	indv_erg_best = {}

	# already_done = np.load("BB_improved3_random_maps_runtime_4_agents.npy",allow_pickle=True).ravel()[0]

	start_pos = np.load(start_pos_file,allow_pickle=True)

	for pbm_file in os.listdir(folder):
		# if pbm_file in already_done.keys():
		# 	continue

		# if pbm_file != "random_map_28.pickle":
		# 	continue

		best_alloc,run_time,per_pruned,indv_erg = branch_and_bound(pbm_file, n_agents, 10, start_pos, random_start=False, scalarize=False)
		print("Best allocation: ", best_alloc)
		print("Runtime: ", run_time)
		run_times[pbm_file] = run_time
		best_allocs[pbm_file] = best_alloc
		per_leaf_prunes[pbm_file] = per_pruned
		indv_erg_best[pbm_file] = indv_erg

		# np.save("BB_improved3_random_maps_runtime_4_agents.npy", run_times)
		# np.save("Best_alloc_BB_improved3_random_maps_4_agents.npy",best_allocs)
		# np.save("per_leaf_pruned_improved3_random_maps_4_agents.npy",per_leaf_prunes)
		# np.save("BB_improved3_random_maps_indv_erg_4_agents.npy", indv_erg_best)


def run_exp_moes(folder,start_pos_file,n_agents):
	run_times = {}
	best_allocs = {}
	indv_ergs = {}
	# start_positions = gen_start_pos(folder,4)
	for pbm_file in os.listdir(folder):
		best_alloc, run_time, indv_erg = main_run_comb_allocation(pbm_file,n_agents,start_pos_file)
		print("Best allocation: ", best_alloc)
		print("Runtime: ", run_time)
		run_times[pbm_file] = run_time
		best_allocs[pbm_file] = best_alloc
		indv_ergs[pbm_file] = indv_erg
		# np.save("Exhaustive_random_maps_runtime_4_agents.npy", run_times)
		# np.save("Exhaustive_random_maps_best_alloc_4_agents.npy",best_allocs)
		# np.save("Exhaustive_random_maps_indv_erg_4_agents.npy",indv_ergs)
		# pdb.set_trace()

	print("Average runtime of exhaustive MOES: ",sum(run_times)/len(run_times))
	print("Runtimes: ", run_times)

def run_exp_eee(folder):
	run_times = []
	for pbm_file in os.listdir(folder):
		
		best_alloc,run_time = main_run_EEE(pbm_file,2)
		print("Best allocation: ", best_alloc)
		print("Runtime: ", run_time)
		run_times.append(run_time)

	print("Average runtime of EEE function allocation: ",sum(run_times)/len(run_times))
	print("Runtimes: ", run_times)

def run_exp_moes_with_heuristic(folder):
	run_times = []
	for pbm_file in os.listdir(folder):
		
		best_alloc,run_time = main_run_comb_with_heuristics(pbm_file,2,0.5)
		print("Best allocation: ", best_alloc)
		print("Runtime: ", run_time)
		run_times.append(run_time)

	print("Average runtime of EEE function allocation: ",sum(run_times)/len(run_times))
	print("Runtimes: ", run_times)

def run_exp_win(folder):
	run_times = []
	for pbm_file in os.listdir(folder):

		best_alloc,run_time = main_run_win_alloc(pbm_file,2)
		print("Best allocation: ", best_alloc)
		print("Runtime: ", run_time)
		# pdb.set_trace()
		run_times.append(run_time)

	print("Average runtime of exhaustive MOES: ",sum(run_times)/len(run_times))
	print("Runtimes: ", run_times)


# import cProfile, pstats, io
# from pstats import SortKey

# pr = cProfile.Profile()
# pr.enable()
# run_exp_bb("./build_prob/random_maps/")
# pr.disable()
# s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--method', type=str, required=True, help="Method to run")
	parser.add_argument('--test_folder', type=str, required=False, help="Folder with test cases", default="./build/test_cases/")
	parser.add_argument('--start_positions',type=str,required=True, help="npy file of agent start positions")
	parser.add_argument('--n_agents',type=int,required=False,default=4)

	args = parser.parse_args()
	folder = args.test_folder
	method = args.method
	start_pos = args.start_positions
	n_agents = args.n_agents 

	if method == "MOES":
		# gen_start_pos(folder,2)
		run_exp_moes(folder,start_pos,n_agents)
	elif method == "window":
		run_exp_win(folder)
	elif method == "BB":
		run_exp_bb(folder,start_pos,n_agents)
	elif method == "EEE":
		run_exp_eee(folder)
	elif method == "MOES_EEE":
		run_exp_moes_with_heuristic(folder)
	else:
		print("Please enter a valid method and folder")

