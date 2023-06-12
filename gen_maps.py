import pickle
import numpy as np
from distributions import gaussianMixtureDistribution
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
    s0 = np.array([0.5,0.5,0])
    pix = 100
    pdfs = []

    for _ in range(n_maps):
        n_peaks = 1 #random.randint(1,3)
        m = []
        c = []
        for _ in range(n_peaks):
            m.append([random.randint(5,80)/100,random.randint(5,80)/100])
            c.append([[random.randint(10,30)/1000,0],[0,random.randint(10,30)/1000]])
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

def map_from_start_pos(pos,pbm_file_name):
    s0 = np.array([0.5,0.5,0])
    pix = 100
    nA = 100
    pdfs = []
    n_peaks = 1
    mu = np.array([[pos[1],pos[0]]])
    cov = np.array([[[random.randint(1,3)/1000,0],[0,random.randint(1,3)/1000]]])
    pdf = gaussianMixtureDistribution(n_peaks, pix, mus=mu, covs=cov)
    pdfs.append(pdf)

    dic = dict()
    dic["s0"] = s0
    dic["nA"] = nA
    dic["pdfs"] = pdfs
    dic["nPixel"] = pix

    SavePickle(dic, pbm_file_name)

if __name__ == "__main__":
	nA = 100
	n_examples = 1
	for i in range(n_examples):
		pbm_file = "build_prob/unimodal_" + str(i) + ".pickle"
		n_maps = 1 #random.randint(4,7)
		GenMOESProblemFourier(nA, pbm_file, n_maps)

