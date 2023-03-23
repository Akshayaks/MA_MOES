import numpy as np
import pdb
import common

import ergodic_metric
from utils import *
from explicit_allocation import *

import matplotlib.pyplot as plt
import ergodic_metric
from distributions import gaussianMixtureDistribution
from gen_maps import *

np.random.seed(100)

"""
Branch and bound using clusters of maps based on the norm of the difference in their fourier 
coefficients. The agents are then allotted to one cluster instead of being allotted to a set
of maps. Every level of the tree will correspond to the assignment of an agent to one cluster.
Thus the tree will have a depth equal to the number of agents.
"""

def generate_map(nA,pbm_file_name):
    s0 = np.array([0.5,0.5,0.0])
    pix = 100
    # save_dir = "/build/instances/"

    n_maps = 3
    pdfs = []
    m = [[[0.05,0.05]],[[0.9,0.9]],[[0.05,0.05]]]
    c = [[[[0.02,0],[0,0.02]]],[[[0.02,0],[0,0.02]]],[[[0.012,0],[0,0.012]]]]

    for i in range(n_maps):
        n_peaks = 1
        mu = np.array(m[i])
        cov = np.array(c[i])
        pdf = gaussianMixtureDistribution(n_peaks, pix, mus=mu, covs=cov)
        pdfs.append(pdf)

    dic = dict()
    dic["s0"] = s0
    dic["nA"] = nA
    dic["pdfs"] = pdfs
    # Erg_obj.phik_array
    dic["nPixel"] = pix

    SavePickle(dic, pbm_file_name)

if __name__ == "__main__":
    pbm_file = "build_prob/instances/unimodal_different_maps.pickle"
    generate_map(100,pbm_file)
    pbm = common.LoadProblem(pbm_file, 1, pdf_list=True)
    display_map(pbm,pbm.s0)

    print("Computing FCs for the maps:")

    n_scalar = 10
    FC = []

    for pdf in pbm.pdfs:
        EC = ergodic_metric.ErgCalc(pdf.flatten(),1,pbm.nA,n_scalar,pbm.pix)
        FC.append(EC.phik)
    
    print("Fourier Coefficients:\n")
    for i in range(len(FC)):
        sum = 0
        for j in range(len(FC[i])):
            sum += (FC[i][j] - FC[0][j])*(FC[i][j] - FC[0][j])
        # print("Total abs diff to FC1: ", np.sqrt(sum))   
    print("Norm of the diff in the FC12: ", np.linalg.norm(FC[1] - FC[0]))
    print("Norm of the diff in the FC23: ", np.linalg.norm(FC[2] - FC[0]))

    pdf_new = np.array(pbm.pdfs[2])
    sum_pt = 0

    for i in range(100):
        for j in range(100):
            pdf_new[i][j] = 1 - pdf_new[i][j]
            sum_pt += pdf_new[i][j]
    
    pdf_new /= sum_pt
    pbm.pdfs.append(pdf_new)
    display_map(pbm,pbm.s0)
    EC = ergodic_metric.ErgCalc(pdf_new.flatten(),1,pbm.nA,n_scalar,pbm.pix)
    FC_new = EC.phik
    print("New diff: ", np.linalg.norm(FC_new - FC[0]))


    # old_FC = FC[2]

    # print("Perturning the FC of 2 a little bit")
    # FC[2] = np.array(FC[2])
    # FC[2][15] += 0.01
    # FC[2][20] += 0.01
    # FC[2][25] -= 0.01
    # FC[2][30] -= 0.01

    # print("Norm of the diff in FC2oldFC: ", np.linalg.norm(FC[2] - old_FC))


    # best_alloc = np.load("./results_canebrake/BB_opt_Best_alloc_4_agents.npy",allow_pickle=True)
    # best_alloc = best_alloc.ravel()[0]
    # n_agents = 4
    # max_diff = []
    # max_diff_sim = []

    # for pbm_file in best_alloc.keys():
    #     alloc = best_alloc[pbm_file]
    #     print("Best allocation: ", alloc)
    #     if(len(alloc) < n_agents + 1):
    #         print("******Incomplete allocation**********")
    #         continue
    #     pbm_file_complete = "./build_prob/random_maps/" + pbm_file
    #     pbm = common.LoadProblem(pbm_file_complete, 4, pdf_list=True)
    #     FC = []
    #     for pdf in pbm.pdfs:
    #         EC = ergodic_metric.ErgCalc(pdf.flatten(),1,pbm.nA,n_scalar,pbm.pix)
    #         FC.append(EC.phik)
    #     for i in range(n_agents):
    #         a = alloc[i]
    #         print("Allocation: ", a)
    #         for j in range(len(pbm.pdfs)):
    #             if j not in a:
    #                 for k in a:
    #                     print("Norm diff of %d, %d: ", j,k,np.linalg.norm(FC[j] - FC[k]))
    #     pdb.set_trace()

