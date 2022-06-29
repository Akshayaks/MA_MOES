import numpy as np 
import matplotlib.pyplot as plt

import common as cm

import pickle
import random
import pdb

from distributions import gaussianMixtureDistribution

def SavePickle(obj, file_path):
    """
    Save a serie of tests as pickle file.
    """
    pickle_out = open(file_path,"wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()
    return

max_maps = 10 #Number of maps starting from 2 to 10
ex_per_no = 5 #Five example cases for each .i.e total of 5*9 = 45 example cases

for n in np.arange(2,11):
	print("Number of maps: ", n)
	for i in range(ex_per_no):

		print("Experiment number: ", i)

		pbm_file_name = "./build/test_cases/"+str(n)+"_maps_example_"+str(i)+".pickle"
		pix = 100 # number of pixels for plotting
		k = 0
		s0 = np.array([.5, .5, 0]) # initial state of robot, add orientation 0
		nA = 100 # number of actions/time horizon
		save_dir = "build/test_cases/"

		pdfs = []

		for j in range(n): #generate n maps
			no_peaks = random.sample(range(5, 10), 1)[0]
			print("Number of peaks: ", no_peaks)
			mu_centers = np.random.uniform(low=0.1, high=0.9, size=(no_peaks*2,))
			cov_val = np.random.uniform(low=0.005, high=0.03, size=(no_peaks*2,))
			
			mu = []
			cov = []
			for m in np.arange(0,len(mu_centers),2):
				mu.append([mu_centers[m],mu_centers[m+1]])
				cov.append([[cov_val[m],0],[0,cov_val[m+1]]])

			pdf = gaussianMixtureDistribution(no_peaks, pix, mus=np.array(mu), covs=np.array(cov))
			x = np.linspace(0,100,num=100)
			y = np.linspace(0,100,num=100)
			X,Y = np.meshgrid(x,y)

			fig, axs = plt.subplots(2, 2)
			axs[0,0].contourf(X, Y, pdf, levels=np.linspace(np.min(pdf), np.max(pdf),100), cmap='gray')
			axs[0,0].set_title('Info Map')
			# plt.pause(1)
			# plt.show()
			pdfs.append(pdf)


		dic = dict()
		dic["s0"] = s0
		dic["nA"] = nA
		dic["pdfs"] = pdfs
		dic["nPixel"] = pix

		SavePickle(dic, pbm_file_name)
