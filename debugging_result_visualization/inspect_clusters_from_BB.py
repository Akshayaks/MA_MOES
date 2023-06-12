import numpy as np
import matplotlib.pyplot as plt

import pdb
from utils import *
import ergodic_metric
import common

"""
This code looks at resulting best allocs from BB and computes the similarity between maps in them
"""

def compute_similarity(pdfs):
	FC = []
	for pdf in pdfs:
		EC = ergodic_metric.ErgCalc(pdf.flatten(),1,100,10,100)
		FC.append(EC.phik)
	norm_matrix = np.zeros((len(FC),len(FC)))
	max_norm = -500
	min_norm = 10000
	for i in range(len(FC)):
		for j in np.arange(0,i):
			norm = np.linalg.norm(FC[i] - FC[j])
			norm_matrix[i][j] = norm
			if norm > max_norm:
				max_norm = norm
			elif norm < min_norm:
				min_norm = norm
	# print("The norm of difference in FC of maps in this cluster: ", norm_matrix)
	# print("The max and min norm are: ", max_norm, min_norm)
	# pdb.set_trace()
	return norm_matrix,min_norm,max_norm

if __name__ == "__main__":
    print("This program inspects the similarity between maps grouped together by BB")
    best_alloc = np.load("./results_canebrake/BB_opt_Best_alloc_4_agents.npy",allow_pickle=True)
    best_alloc_sim = np.load("./results_canebrake/BB_similarity_clustering_best_alloc_4_agents.npy",allow_pickle=True)
    best_alloc = best_alloc.ravel()[0]
    best_alloc_sim = best_alloc_sim.ravel()[0]
    n_agents = 4
    max_diff = []
    max_diff_sim = []

    for pbm_file in best_alloc.keys():
        alloc = best_alloc[pbm_file]
        print("Best allocation: ", alloc)
        if(len(alloc) < n_agents + 1):
            print("******Incomplete allocation**********")
            continue
        alloc_sim = best_alloc_sim[pbm_file]
        pbm_file_complete = "./build_prob/random_maps/" + pbm_file
        pbm = common.LoadProblem(pbm_file_complete, 4, pdf_list=True)
        for i in range(n_agents):
            a = alloc[i]
            print("Current cluster: ", a)
            if len(a) > 1:
                pdf_list = [pbm.pdfs[mi] for mi in a]
            else:
                pdf_list = [pbm.pdfs[a[0]]]
            norm_matrix, min_norm, max_norm = compute_similarity(pdf_list)
            if max_norm >= 0:
                max_diff.append(max_norm)
            a = alloc_sim[i]
            print("Current cluster: ", a)
            if len(a) > 1:
                pdf_list = [pbm.pdfs[mi] for mi in a]
            else:
                pdf_list = [pbm.pdfs[a[0]]]
            norm_matrix, min_norm, max_norm = compute_similarity(pdf_list)
            if max_norm >= 0:
                max_diff_sim.append(max_norm)
    x_axis = np.arange(0,len(max_diff))
    x_axis_sim = np.arange(0,len(max_diff_sim))
    plt.plot(x_axis,max_diff)
    # plt.plot(x_axis_sim,max_diff_sim)
    # plt.legend(["BB", "BB_sim"])
    plt.title("Norm of the difference in the FC of maps assigned to one agent")
    plt.xlabel("test Number")
    plt.ylabel("Maximum norm of the difference in FC of maps assigned to one agent")
    plt.show()
