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

def run_exp_bb(folder):
	run_times = {}
	best_allocs = {}
	per_leaf_prunes = {}
	# start_positions = gen_start_pos(folder,2)
	for pbm_file in os.listdir(folder):
		best_alloc,run_time,per_pruned = branch_and_bound(pbm_file, 2, 10, random_start=False, start_pos_file="start_pos_random_2_agents.npy", scalarize=False)
		print("Best allocation: ", best_alloc)
		print("Runtime: ", run_time)
		run_times[pbm_file] = run_time
		best_allocs[pbm_file] = best_alloc
		per_leaf_prunes[pbm_file] = per_pruned
		np.save("BB_random_maps_runtime_2_agents.npy", run_times)
		np.save("Best_alloc_BB_random_maps_2_agents.npy",best_allocs)
		np.save("per_leaf_pruned_random_maps_2_agents.npy",per_leaf_prunes)

	print("Average runtime of BB: ",sum(run_times)/len(run_times))
	print("Runtimes: ", run_times)


def run_exp_moes(folder):
	run_times = {}
	best_allocs = {}
	start_positions = gen_start_pos(folder,2)
	for pbm_file in os.listdir(folder):
		best_alloc,run_time = main_run_comb_allocation(pbm_file,2)
		print("Best allocation: ", best_alloc)
		print("Runtime: ", run_time)
		run_times[pbm_file] = run_time
		best_allocs[pbm_file] = best_alloc
		np.save("MOES_random_maps_runtime_2_agents.npy", run_times)
		np.save("Best_alloc_MOES_random_maps_2_agents.npy",best_allocs)

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


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--method', type=str, required=True, help="Method to run")
	parser.add_argument('--test_folder', type=str, required=True, help="Folder with test cases", default="./build/test_cases/")

	args = parser.parse_args()
	folder = args.test_folder
	method = args.method 

	if method == "MOES":
		# gen_start_pos(folder,2)
		run_exp_moes(folder)
	elif method == "window":
		run_exp_win(folder)
	elif method == "BB":
		run_exp_bb(folder)
	elif method == "EEE":
		run_exp_eee(folder)
	elif method == "MOES_EEE":
		run_exp_moes_with_heuristic(folder)
	else:
		print("Please enter a valid method and folder")

