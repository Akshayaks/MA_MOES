import numpy as np
import matplotlib.pyplot as plt

import pdb
from utils import *
import ergodic_metric
import common

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

from sklearn.datasets import make_blobs
import pandas as pd
import hdbscan

def k_means_clustering(pbm,n_agents,n_scalar):
    pdfs = pbm.pdfs
    data = [pdf.flatten() for pdf in pdfs]
    data = []
    # n_scalar = 10
    for pdf in pdfs:
        EC = ergodic_metric.ErgCalc(pdf.flatten(),1,pbm.nA,n_scalar,pbm.pix)
        data.append(EC.phik*np.sqrt(EC.lamk))
    # print("Sclarizing the first two information maps")
    # pdf_scala = (pdfs[0] + pdfs[1])/2
    # EC = ergodic_metric.ErgCalc(pdf_scala.flatten(),1,pbm.nA,n_scalar,pbm.pix)
    # phi_scala = EC.phik
    # print("phi1: ", data[0][:10])
    # print("phi2: ", data[1][:10])
    # print("phi_scala: ", phi_scala[:10])
    # pdb.set_trace()
    cost =[]
    s_score = []
    for i in range(1, len(pdfs)+1):
        print("Number of clusters: ", i)
        KM = KMeans(n_clusters = i, max_iter = 500, random_state=0)
        KM.fit(data)
        # pdb.set_trace()
        # calculates squared error
        # for the clustered points
        cluster_labels = KM.labels_
        print("labels: ", cluster_labels)
        # pdb.set_trace()
        if len(KM.cluster_centers_) == 1 or len(KM.cluster_centers_) == len(pdfs):
            s_score.append(-1)
        else:
            silhouette_avg = silhouette_score(data,cluster_labels)
            s_score.append(silhouette_avg)
        cost.append(KM.inertia_)
        print("\nCost for this clustering: ", cost)
        print("\nSil score: ", s_score)
        #plot the cost against K values
    x = np.arange(1,len(pdfs)+1)
    kn = KneeLocator(x, cost, curve='convex', direction='decreasing', online=True, S=1)
    print("knee: ", kn.knee)
    print("knee_s_score: ", np.argmax(s_score)+1)
    n_clusters = np.argmax(s_score)+1
    # pdb.set_trace()
    ##### For now we want the n_clusters >= n_agents ########
    if n_clusters < n_agents:
        print("\n*************Forcing number of clusters to be equal to number of agents!!********\n")
        n_clusters = n_agents #len(pdfs)

    # plt.plot(range(1, len(pdfs)+1), cost, color ='g', linewidth ='3')
    # plt.plot(range(1, len(pdfs)+1), s_score, color ='r', linewidth ='3')
    # plt.xlabel("Value of K")
    # plt.ylabel("Squared Error (Cost)")
    # plt.show() # clear the plot
    Kmean = KMeans(n_clusters=n_clusters)
    Kmean.fit(data)
    print("\nThe cluster labels are: ", Kmean.labels_)
    # pdb.set_trace()
    clusters = [[] for _ in range(n_clusters)]
    labels = Kmean.labels_
    for idx,l in enumerate(labels):
        clusters[l].append(idx)

    print("\nFinal clusters are: ", clusters)
    # pdb.set_trace()

    return clusters

def HDBScan(pbm,n_scalars,n_agents):
    pdfs = pbm.pdfs
    data = [pdf.flatten() for pdf in pdfs]
    data = []
    # n_scalar = 10
    for pdf in pdfs:
        EC = ergodic_metric.ErgCalc(pdf.flatten(),1,pbm.nA,n_scalar,pbm.pix)
        data.append(EC.phik)
    blobs, labels = make_blobs(n_samples=len(data), n_features=n_scalars*n_scalars)
    pd.DataFrame(blobs).head()
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    gen_min_span_tree=False, leaf_size=40,
    metric='euclidean', min_cluster_size=2, min_samples=None, p=None)
    clusterer.fit(blobs)
    labels = clusterer.labels_
    print("Cluster labels: ", clusterer.labels_)
    n_clusters = np.unique(labels)
    # pdb.set_trace()
    # HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    # gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None),
    # metric='euclidean', min_cluster_size=5, min_samples=None, p=None)

    # clusters = [[] for _ in range(n_clusters)]
    # # labels = Kmean.labels_
    # for idx,l in enumerate(labels):
    #     clusters[l].append(idx)

    # print("\nFinal clusters are: ", clusters)
    # # pdb.set_trace()

    # return clusters

def same_alloc(a1,a2):
    same_clustering = True
    for c in a1.values():
        if c == []:
            continue
        if tuple(c) not in a2.values():
            same_clustering = False
    return same_clustering


if __name__ == "__main__":
    # best_alloc = np.load("./results_canebrake/BB_opt_random_maps_sparse_Best_alloc_4_agents.npy",allow_pickle=True)
    best_alloc = np.load("./results_canebrake/BB_opt_Best_alloc_4_agents.npy",allow_pickle=True)
    best_alloc = best_alloc.ravel()[0]

    # indv_erg_BB = np.load("./results_canebrake/BB_opt_random_maps_sparse_indv_erg_4_agents.npy",allow_pickle=True)
    indv_erg_BB = np.load("./results_canebrake/BB_opt_indv_erg_4_agents.npy",allow_pickle=True)
    indv_erg_BB = indv_erg_BB.ravel()[0]

    # best_alloc_sim = np.load("BB_similarity_clustering_sparse_maps_best_alloc_4_agents.npy",allow_pickle=True)
    best_alloc_sim = np.load("BB_similarity_clustering_random_maps_best_alloc_4_agents.npy",allow_pickle=True)
    best_alloc_sim = best_alloc_sim.ravel()[0]

    # indv_erg_sim = np.load("BB_similarity_clustering_sparse_maps_indv_erg_4_agents.npy",allow_pickle=True)
    indv_erg_sim = np.load("BB_similarity_clustering_random_maps_indv_erg_4_agents.npy",allow_pickle=True)
    indv_erg_sim = indv_erg_sim.ravel()[0]

    n_agents = 4
    n_scalar = 10
    cnt = 0

    file1 = open("similarity_based_BandB.txt","w")
    run_times = {}
    best_allocs = {}
    indv_erg_best = {}

    # best_alloc = np.load("./results_canebrake/BB_opt_Best_alloc_4_agents.npy",allow_pickle=True)
    # best_alloc = best_alloc.ravel()[0]

    num = 0
    same = 0

    for file in os.listdir("build_prob/random_maps/"):
        # pbm_file = "build_prob/random_maps/"+file

        # print("\nFile: ", file)

        # problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)

        # if len(problem.pdfs) < 4 or len(problem.pdfs) > 7:
        #     continue

        if file not in best_alloc_sim.keys() or file not in best_alloc.keys():
            continue

        num += 1

        # start_pos = np.load("start_pos_ang_random_4_agents.npy",allow_pickle=True)
        # problem.s0 = start_pos.item().get(file)
        # clusters = k_means_clustering(problem,n_agents,n_scalar)
        # HDB_clusters = HDBScan(problem,n_scalar,n_agents)
        print("Similarity clustering allocation: ", best_alloc_sim[file])
        print("BB opt allocation: ", best_alloc[file])
        if same_alloc(best_alloc_sim[file],best_alloc[file]):
            same += 1
        else:
            print("Minmax BB: ", max(indv_erg_BB[file]))
            print("Minmax sim: ", max(indv_erg_sim[file]))
            print("Difference in minmax metric: ", np.abs(max(indv_erg_BB[file]) - max(indv_erg_sim[file])))
            pdb.set_trace()
    print("Total number of tests: ", num)
    print("Total number of same clustering: ", same)