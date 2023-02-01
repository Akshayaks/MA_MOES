import pickle
import numpy as np
import copy

import pymoo.model.problem as pmo_p

import ergodic_metric
from distributions import gaussianMixtureDistribution

import matplotlib.pyplot as plt
import jax.numpy as jnp
import random

def SavePickle(obj, file_path):
    """
    Save a serie of tests as pickle file.
    """
    pickle_out = open(file_path,"wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()
    return

def GenMOESProblemFourier(nA, pbm_file_name,n_maps):
    s0 = np.array([0.5,0.5])
    pix = 100
    # save_dir = "/build/instances/"

    # n_maps = random.randint(1, 5)
    pdfs = []

    for _ in range(n_maps):
        n_peaks = random.randint(1,6)
        m = []
        c = []
        for j in range(n_peaks):
            m.append([random.randint(5,80)/100,random.randint(5,80)/100])
            c.append([[random.randint(1,5)/100,0],[0,random.randint(1,5)/100]])
        mu = np.array(m)
        cov = np.array(c)
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
	nA = 100
	n_examples = 100
	for i in range(n_examples):
		pbm_file = "build_prob/random_maps_20/random_map_" + str(i) + ".pickle"
		n_maps = 20
        # random.randint(3,11)
		GenMOESProblemFourier(nA, pbm_file, n_maps)

