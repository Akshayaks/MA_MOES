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
from scipy.signal import find_peaks
import ergodic_metric
from utils import *
from explicit_allocation import *
import math
import time
import json

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from kneed import KneeLocator
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

np.random.seed(100)

"""
Cluster information maps based on norm of the difference in the fourier coefficients. Assign agents to clusters based on the
distance of the agent from the centroid of the clusters. Again we ensure the number of clusters is >= number of agents 
"""


def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks


def k_means_clustering(pbm,n_agents,n_scalar):
	
	pdfs = pbm.pdfs
	data = [pdf.flatten() for pdf in pdfs]
	data = []
	for pdf in pdfs:
		EC = ergodic_metric.ErgCalc(pdf.flatten(),1,pbm.nA,n_scalar,pbm.pix)
		data.append(EC.phik)
	cost =[]
	for i in range(1, len(pdfs)+1):
		KM = KMeans(n_clusters = i, max_iter = 500)
		KM.fit(data)
		# calculates squared error
		# for the clustered points
		cost.append(KM.inertia_*100000)
		print("\nCost for this clustering: ", cost)
	x = np.arange(1,len(pdfs)+1)
	kn = KneeLocator(x, cost, curve='convex', direction='decreasing')
	print("knee: ", kn.knee)

	##### For now we want the n_clusters >= n_agents ########
	if kn.knee:
		if kn.knee < n_agents:
			print("\n*************Forcing number of clusters to be equal to number of agents!!********\n")
			n_clusters = n_agents
		else:
			n_clusters = kn.knee
	else:
		n_clusters = n_agents

	# plt.plot(range(1, len(pdfs)+1), cost, color ='g', linewidth ='3')
	# plt.xlabel("Value of K")
	# plt.ylabel("Squared Error (Cost)")
	# plt.show() # clear the plot
	Kmean = KMeans(n_clusters=n_clusters)
	Kmean.fit(data)
	print("\nThe cluster labels are: ", Kmean.labels_)
	clusters = [[] for _ in range(n_clusters)]
	labels = Kmean.labels_
	for idx,l in enumerate(labels):
		clusters[l].append(idx)
	
	print("\nFinal clusters are: ", clusters)
	return clusters

def distance_allocate(pbm, clusters, peak_centers, n_agents):
    # start_time = time.time()
    # best_alloc = {}
    # indv_erg = []
    agent_assigned = np.zeros(n_agents)

    agent_scores = np.zeros((n_agents,len(clusters)))

    for i in range(len(clusters)):
        for j in range(n_agents):
            ## In start pos: -> in first coordinate, | is second. In peak | is first coordinate
            d = (pbm.s0[3*j+1]*100 - peak_centers[i][0])**2 + (pbm.s0[3*j]*100 - peak_centers[i][1])**2
            d = math.sqrt(d)
            agent_scores[j][i] = d
    
    print("Agent scores: ", agent_scores)
    # pdb.set_trace()

    #Calculate distance of agent to cluster peaks centroid

    
    allocation = {}

    if n_agents == len(clusters):
        agents_assigned = np.zeros(len(clusters))
        print("No = Na")
        ## Here we are not iterating though clusters and are doing it through agents instead
        for i in range(len(clusters)):
            k = 0
            erg = sorted([a[i] for a in agent_scores])
            found = False
            while not found:
                idx = [a[i] for a in agent_scores].index(erg[k])
                if agent_assigned[idx]:
                    k += 1
                else:
                    allocation[idx] = [i]
                    found = True
                    agent_assigned[idx] = 1
            # print("\nClusters assigned: ", agent_assigned) 
    else:
        print("No > Na")
        agents_assigned = np.zeros(n_agents)
        for j in range(n_agents):
            allocation[j] = []
        # print("Allocation initialized")
        for i in range(len(clusters)):
            k = 0
            erg = sorted([a[i] for a in agent_scores])
            # print("agent_scores: ",erg)
            found = False
            while not found:
                idx = [a[i] for a in agent_scores].index(erg[k])
                zero_map_agents = list(agents_assigned).count(0)
                if (agents_assigned[idx] > 0 and len(clusters)-i == zero_map_agents):
                    k += 1
                else:
                    allocation[idx] = allocation[idx] + [i]
                    found = True
                    agents_assigned[idx] += 1
                    # print("allocation so far: ", allocation)
    print("The final allocations are as follows: ", allocation)

    # for i in range(len(clusters)):
    #     distances = []
    #     assigned = False
    #     for j in range(n_agents):
    #         ## In start pos: -> in first coordinate, | is second. In peak | is first coordinate
    #         d = (pbm.s0[3*j+1]*100 - peak_centers[i][0])**2 + (pbm.s0[3*j]*100 - peak_centers[i][1])**2
    #         d = math.sqrt(d)
    #         distances.append(d)
    #     info = sorted(distances)
    #     k = 0 
        
    #     while not assigned:
    #         agent_idx = distances.index(info[k])
    #         if agent_assigned[agent_idx] == 0:
    #             agent_assigned[agent_idx] = 1
    #             best_alloc[i] = agent_idx
    #             assigned = True
    #         else:
    #             k = k + 1
    #     print("Best allocation: ", best_alloc)
    # pdb.set_trace()       
    # runtime = time.time() - start_time
    return allocation
	
if __name__ == "__main__":
    n_agents = 3
    n_scalar = 10
    cnt = 0

    # file1 = open("distance_allocate_3_agents.txt","w")
    run_times = {}
    best_allocs = {}
    indv_erg_best = {}

    for file in os.listdir("build_prob/random_maps/"):
        start_time = time.time()
        pbm_file = "build_prob/random_maps/"+file

        print("\nFile: ", file)

        # if file != "random_map_28.pickle":
        #      continue

        problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)


        if len(problem.pdfs) < n_agents:
            continue

        start_pos = np.load("start_pos_random_3_agents.npy",allow_pickle=True)
        problem.s0 = start_pos.item().get(file)

        print(problem.s0)
        display_map(problem,problem.s0)

        if len(problem.pdfs) == n_agents:
            clusters = [[i] for i in range(n_agents)]
        else:
            clusters = k_means_clustering(problem,n_agents,n_scalar)
        print("The clusters are: ", clusters)

        peak_centers = {}

        for i,ci in enumerate(clusters):
            pdf = np.zeros((100,100))
            for mi in ci:
                 pdf += problem.pdfs[mi]
            pdf = pdf/len(ci)
            peaks = detect_peaks(pdf)
            n_peaks = np.where(peaks==True)
            print("Peaks: ", n_peaks)
            print("Length of peaks: ", len(n_peaks),len(n_peaks[0]))
            px = 0
            py = 0
            if len(n_peaks[0]) > 1:
                 print("\nMultiple peaks")
                 print("peaks[0]: ", n_peaks[0],sum(n_peaks[0]),len(n_peaks[0]))
                 print("peaks[1]: ", n_peaks[1],sum(n_peaks[1]),len(n_peaks[1]))
                 px = sum(n_peaks[0])/len(n_peaks[0])
                 py = sum(n_peaks[1])/len(n_peaks[1])
                 peak_centers[i] = (px,py)
            else:
                 print("\nSingle peak")
                 px = n_peaks[0]*1
                 py = n_peaks[1]*1
                 peak_centers[i] = (px,py)
            fig, axs = plt.subplots(1,2,figsize=(5,5))
            axs[0].imshow(pdf)
            axs[1].imshow(peaks)
            
            for ai in range(n_agents):
                print("start: ", problem.s0[ai*3]*100, problem.s0[ai*3+1]*100)
                axs[0].plot(problem.s0[ai*3]*100,problem.s0[ai*3+1]*100, marker="o", markersize=5, markerfacecolor='red', markeredgecolor='black')
            plt.show()
            # print("\nPeak coordinate: ", px, py)
            # pdb.set_trace()
        print("\nClusters peak coordinates: ", peak_centers)
        print("Agent start positions: ", problem.s0)
        # pdb.set_trace()

        incumbent = distance_allocate(problem,clusters,peak_centers,n_agents)

        incumbent_erg = np.zeros(len(problem.pdfs))

        for k,v in incumbent.items():
            # print("\nagent: ", k)
            # print("\ncluster: ", v)
            # print("\nmaps: ", clusters[v[0]])

            pdf = np.zeros((100,100))
            n_maps = 0
            for a in v:
                for mapi in clusters[a]:
                    pdf += problem.pdfs[mapi]
                    n_maps += 1
            pdf = (1/n_maps)*pdf

            # print("v: ", v)
            # if len(v) > 1:
            # 	pdf = np.zeros((100,100))
            # 	length = len(pdf_list)
            # 	for a in v:
            # 		pdf += (1/length)*pdf_list[a]
            # 	# scalarize_minmax([pdf_list[a] for a in v],problem.s0[k*3:k*3+3],problem.nA)
            # else:
            # 	pdf = pdf_list[v[0]]

            pdf = np.asarray(pdf.flatten())
            
            #Just run ergodicity optimization for fixed iterations and see which agent achieves best ergodicity in that time
            control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)
            print("\nmap erg: ", erg[-1])
            
            for p in v:
                for mapi in clusters[p]:
                    print("\nmapi: ", mapi)
                    pdf_indv = np.asarray(problem.pdfs[mapi].flatten())
                    EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
                    incumbent_erg[mapi] = EC.fourier_ergodic_loss(control,problem.s0[3*k:3+3*k])
                    print("\nincum erg: ", incumbent_erg[mapi])

        upper = max(incumbent_erg)
        print("Incumbent allocation: ", incumbent)
        print("Incumber Ergodicities: ", incumbent_erg)
        print("Initial Upper: ", upper)
        
        

        best_alloc = {}
        for i in range(n_agents):
            best_alloc[i] = []
        
        for i in range(n_agents):
            for c in incumbent[i]:
                best_alloc[i] = best_alloc[i] + clusters[c]

        print("\nBest allocation is: ", best_alloc)
        print("\nBest Individual ergodicity: ", incumbent_erg)            
        # pdb.set_trace()

        run_times[file] = time.time() - start_time
        best_allocs[file] = best_alloc
        indv_erg_best[file] = incumbent_erg

        # np.save("dist_3_agents_runtime.npy", run_times)
        # np.save("dist_3_agents_best_alloc.npy", best_allocs)
        # np.save("dist_3_agents_indv_erg.npy", indv_erg_best)

    #     file1.write(file)
    #     file1.write("\n")
    #     file1.write(json.dumps(incumbent))
    #     file1.write("\n")
    #     file1.write("clusters: ")
    #     for c in clusters:
    #         for ci in c:
    #             file1.write(str(ci))
    #             file1.write(", ")
    #         file1.write("; ")
    #     file1.write("\n")

    # file1.close()






