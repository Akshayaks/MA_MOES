import numpy as np
import os
import common
from utils import *
import ergodic_metric
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import pdb

def get_vectors(pdf_list):
    FC = []
    for pdf in pdf_list:
        EC = ergodic_metric.ErgCalc(pdf.flatten(),1,100,30,100)
        FC.append(EC.phik*EC.lamk*EC.lamk)
    return FC

def k_means_clustering(pbm,data,n_agents,n_scalar):
    pdfs = pbm.pdfs
    # data = [pdf.flatten() for pdf in pdfs]
    # data = []
    # # n_scalar = 10
    # for pdf in pdfs:
    #     EC = ergodic_metric.ErgCalc(pdf.flatten(),1,pbm.nA,n_scalar,pbm.pix)
    #     data.append(EC.phik)
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
    pdb.set_trace()
    ##### For now we want the n_clusters >= n_agents ########
    if n_clusters < n_agents:
        print("\n*************Forcing number of clusters to be equal to number of agents!!********\n")
        n_clusters = len(pdfs)

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


if __name__ == "__main__":
    best_alloc = np.load("./results_canebrake/BB_opt_Best_alloc_4_agents.npy",allow_pickle=True)
    best_alloc = best_alloc.ravel()[0]
    best_alloc_sim = np.load("./results_canebrake/BB_similarity_clustering_best_alloc_4_agents.npy",allow_pickle=True)
    best_alloc_sim = best_alloc_sim.ravel()[0]
    for pbm_file in os.listdir("./build_prob/random_maps/"):
        pbm_file_complete = "./build_prob/random_maps/" + pbm_file
        pbm = common.LoadProblem(pbm_file_complete, 4, pdf_list=True)

        n_agents = 4

        if len(pbm.pdfs) < n_agents:
            continue

        print("Number of pdfs: ", len(pbm.pdfs))
        FC_vectors = get_vectors(pbm.pdfs)

        FC_vectors = StandardScaler().fit_transform(FC_vectors) # normalizing the features

        feat_cols = ['feature'+str(i) for i in range(FC_vectors.shape[1])]
        normalised_FC = pd.DataFrame(FC_vectors,columns=feat_cols)
        
        pca_fc = PCA(n_components=n_agents)
        principalComponents_fc = pca_fc.fit_transform(FC_vectors)

        clusters = k_means_clustering(pbm,principalComponents_fc,n_agents=4,n_scalar=10)

        alloc = best_alloc[pbm_file]
        print("Allocation: ", alloc)
        if len(alloc) < 4:
            continue
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        colors = ["red", "blue", "green", "black"]
        C = []

        for mi in range(len(pbm.pdfs)):
            for i in range(4):
                if mi in alloc[i]:
                    C.append(colors[i])
                    break

        print(principalComponents_fc)
        x = []
        y = []
        z = []
        for p in principalComponents_fc:
            x.append(p[0])
            y.append(p[1])
            z.append(p[2])
        
        ax.scatter(x,y,z,c=C)
        ax.set_title("BB alloc")
        # plt.show()

        alloc_sim = best_alloc_sim[pbm_file]
        C = []

        for mi in range(len(pbm.pdfs)):
            for i in range(4):
                if mi in alloc_sim[i]:
                    C.append(colors[i])
                    break

        print(principalComponents_fc)
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        x = []
        y = []
        z = []
        for p in principalComponents_fc:
            x.append(p[0])
            y.append(p[1])
            z.append(p[2])
        
        ax.scatter(x,y,z,c=C)
        ax.set_title("Sim alloc")
        plt.show()


        # plt.figure()
        # plt.figure(figsize=(10,10))
        # plt.xticks(fontsize=12)
        # plt.yticks(fontsize=14)
        # plt.xlabel('Principal Component - 1',fontsize=20)
        # plt.ylabel('Principal Component - 2',fontsize=20)
        # plt.title("Principal Component Analysis of FC vectors",fontsize=20)
        # targets = ['Benign', 'Malignant']
        # colors = ['r', 'g']
        # for target, color in zip(targets,colors):
        #     indicesToKeep = breast_dataset['label'] == target
        #     plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1']
        #             , principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

        # plt.legend(targets,prop={'size': 15})






