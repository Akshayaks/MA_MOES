from explicit_allocation import *
import os
import pdb
import jax
import argparse
import common

'''
Iterate through all the example problems in test_cases and execute MOES on them and store the result
Currently doing it just for 2 agents
'''

def run_exp_moes(folder):
	run_times = []
	for pbm_file in os.listdir(folder):
		
		best_alloc,run_time = main_run_comb_allocation(pbm_file,2)
		print("Best allocation: ", best_alloc)
		print("Runtime: ", run_time)
		run_times.append(run_time)

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
  elif method == "EEE":
  	run_exp_eee(folder)
  elif method == "MOES_EEE":
  	run_exp_moes_with_heuristic(folder)
  else:
  	print("Please enter a valid method and folder")

