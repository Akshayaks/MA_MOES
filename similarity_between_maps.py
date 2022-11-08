import numpy as onp
import sys
import os

from jax import vmap, jit, grad
import jax.numpy as np
from jax.lax import scan
from functools import partial
import pdb
from sklearn.preprocessing import normalize
import common
import scalarize
from ergodic_coverage import ErgCover

import scipy.stats
import ergodic_metric
from utils import *
import math

from sklearn.cluster import KMeans

np.random.seed(100)

def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = p.flatten()
    q = q.flatten()

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance

def ergodic_similarity(problem, n_scalar): #n_scalar -> num of fourier coefficients
	pdf1 = problem.pdfs[0].flatten()
	pdf2 = problem.pdfs[1].flatten()

	EC1 = ergodic_metric.ErgCalc(pdf1,1,problem.nA,n_scalar,problem.pix)
	EC2 = ergodic_metric.ErgCalc(pdf1+0.9,1,problem.nA,n_scalar,problem.pix)

	distance = np.sum(EC1.lamk*np.square(EC1.phik - EC2.phik))
	return distance

def k_means_clustering(pdfs,k):
	Kmean = KMeans(n_clusters=k)
	data = [pdf.flatten() for pdf in pdfs]
	Kmean.fit(data)
	print("\nThe cluster labels are: ", Kmean.labels_)
	
if __name__ == "__main__":
	n_agents = 2
	n_scalar = 10

	for file in os.listdir("build_prob/test_cases/"):
		pbm_file = "build_prob/test_cases/"+file

		print("\nFile: ", file)

		problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)

		problem.nA = 10
		nA = problem.nA

		problem.s0 = np.array([0.1,0.1,0,0.6,0.7,0])
		# random_start_pos(n_agents)

		print("Agent start positions allotted:", problem.s0)

		display_map(problem,problem.s0)
		# p = np.array([[1,5,6,2,0,3],
		# 	[4,6,7,1,2,5],
		# 	[3,2,1,5,4,0],
		# 	[2,4,0,1,3,5],
		# 	[0,2,8,5,3,1]])
		# q = np.array([[2,6,7,3,1,4],
		# 	[5,7,8,2,3,6],
		# 	[3,2,1,5,4,0],
		# 	[2,4,0,1,3,5],
		# 	[0,2,8,5,3,1]])

		# print("\nDistance between map 1 and 2 using Jenson Shanon Distance: ", jensen_shannon_distance(p,q))
		# print("\nDistance between map 1 and 2 using ergodic metric: ", ergodic_similarity(problem, n_scalar))

		k_means_clustering(problem.pdfs,n_agents)


