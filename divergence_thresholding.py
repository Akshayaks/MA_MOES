import matplotlib.pyplot as plt
import numpy as np

import common

import os
from utils import *

def divergence_thresholding(pbm_file):
	n_agents = 1
	problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)
	pdf_list = problem.pdfs
	pdf1 = (pdf_list[0] + pdf_list[1])/2
	pdf2 = (pdf_list[0] + pdf_list[2])/2
	print("Between 01, 02: ", kl_divergence(pdf1,pdf_list[2]))
	# print("Between 2, 12: ", kl_divergence(pdf_list[2],pdf2))
	for i in range(len(pdf_list)):
		for j in np.arange(0,len(pdf_list)):
			dist = kl_divergence(pdf_list[i],pdf_list[j])
			print("Distance between " + str(i) + " and " + str(j) + ": ",dist)

if __name__ == "__main__":
	pbm_file = "./build_prob/test_cases/3_maps_example_0.pickle"
	divergence_thresholding(pbm_file)
