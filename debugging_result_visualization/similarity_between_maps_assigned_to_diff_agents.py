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
			if norm < min_norm and norm != 0:
				min_norm = norm
	# print("The norm of difference in FC of maps in this cluster: ", norm_matrix)
	# print("The max and min norm are: ", max_norm, min_norm)
	# pdb.set_trace()
	return norm_matrix,min_norm,max_norm

def compute_similarity_dc(pdfs):
    FC = []
    for pdf in pdfs:
        EC = ergodic_metric.ErgCalc(pdf.flatten(),1,100,10,100)
        FC.append(EC.phik)
    norm_matrix = np.zeros(len(FC))
    max_norm = -500
    min_norm = 10000
    for i in np.arange(1,len(FC)):
        print(max_norm, min_norm)
        norm = np.linalg.norm(FC[i] - FC[0])
        norm_matrix[i] = norm
        if norm > max_norm:
            max_norm = norm
        if norm < min_norm:
            min_norm = norm
    # print("The norm of difference in FC of maps in this cluster: ", norm_matrix)
    # print("The max and min norm are: ", max_norm, min_norm)
    # pdb.set_trace()
    return norm_matrix,min_norm,max_norm

if __name__ == "__main__":
    print("This program inspects the similarity between maps grouped together by BB")
    best_alloc = np.load("./results_canebrake/BB_opt_Best_alloc_4_agents.npy",allow_pickle=True)
    best_alloc = best_alloc.ravel()[0]
    n_agents = 4
    max_diff_sc = []
    min_diff_sc = []
    max_diff_dc = []
    min_diff_dc = []

    for pbm_file in best_alloc.keys():
        alloc = best_alloc[pbm_file]
        print("Best allocation: ", alloc)
        if(len(alloc) < n_agents + 1):
            print("******Incomplete allocation**********")
            continue
        pbm_file_complete = "./build_prob/random_maps/" + pbm_file
        pbm = common.LoadProblem(pbm_file_complete, 4, pdf_list=True)
        max_sc_norm = 0
        max_dc_norm = 0
        min_sc_norm = 100
        min_dc_norm = 100
        for i in range(n_agents):
            a = alloc[i]
            pdf_list = [pbm.pdfs[ai] for ai in a]
            norm_matrix, min_norm, max_norm = compute_similarity(pdf_list)

            if max_norm > 0 and max_norm > max_sc_norm:
                max_sc_norm = max_norm
            if min_norm != 10000 and min_norm < min_sc_norm:
                min_sc_norm = min_norm
        max_diff_sc.append(max_sc_norm)
        if min_sc_norm == 100:
            min_diff_sc.append(0)
        else:
            min_diff_sc.append(min_sc_norm)

        for i in range(n_agents):
            a = alloc[i]
            other_maps = []
            for ai in a:
                other_maps.append(pbm.pdfs[ai])
                for mi in range(len(pbm.pdfs)):
                    if mi not in a:
                        other_maps.append(pbm.pdfs[mi])
                norm_matrix, min_norm, max_norm = compute_similarity_dc(other_maps)
                # print(norm_matrix)
                # print(max_norm,min_norm)
                # pdb.set_trace()
                if max_norm > 0 and max_norm > max_dc_norm:
                    max_dc_norm = max_norm
                if min_norm != 10000 and min_norm < min_dc_norm:
                    min_dc_norm = min_norm
            
        max_diff_dc.append(max_dc_norm)
        min_diff_dc.append(min_dc_norm)
        if len(max_diff_dc) > 30:
            break

    print("Max diff same cluster: ", max_diff_sc)
    print("Mx diff different cluster: ", max_diff_dc)
    print("Min diff same cluster: ", min_diff_sc)
    print("Min diff different cluster: ", min_diff_dc)
    x_axis = np.arange(0,len(max_diff_dc))
    plt.plot(x_axis,max_diff_sc)
    plt.plot(x_axis,max_diff_dc)
    plt.plot(x_axis,min_diff_dc)
    plt.plot(x_axis,min_diff_sc)
    # plt.plot(x_axis,min_diff)
    plt.legend(["Max diff same cluster", "Max diff different cluster", "Min diff different cluster", "Min diff same cluster"])
    plt.title("Norm of the difference in the FC of maps assigned to same and different agents")
    plt.xlabel("test Number")
    plt.ylabel("Norm of the difference in FC")
    plt.show()
