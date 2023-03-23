import numpy as np
import os
import common
from utils import *
import ergodic_metric
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_vectors(pdf_list):
    FC = []
    for pdf in pdf_list:
        EC = ergodic_metric.ErgCalc(pdf.flatten(),1,100,30,100)
        FC.append(EC.phik)
    return FC


if __name__ == "__main__":
    best_alloc = np.load("./results_canebrake/BB_opt_Best_alloc_4_agents.npy",allow_pickle=True)
    best_alloc = best_alloc.ravel()[0]
    best_alloc_sim = np.load("./results_canebrake/BB_similarity_clustering_best_alloc_4_agents.npy",allow_pickle=True)
    best_alloc_sim = best_alloc_sim.ravel()[0]
    for pbm_file in os.listdir("./build_prob/random_maps/"):
        pbm_file_complete = "./build_prob/random_maps/" + pbm_file
        pbm = common.LoadProblem(pbm_file_complete, 4, pdf_list=True)

        FC_vectors = get_vectors(pbm.pdfs)

        FC_vectors = StandardScaler().fit_transform(FC_vectors) # normalizing the features

        feat_cols = ['feature'+str(i) for i in range(FC_vectors.shape[1])]
        normalised_FC = pd.DataFrame(FC_vectors,columns=feat_cols)
        
        pca_fc = PCA(n_components=3)
        principalComponents_fc = pca_fc.fit_transform(FC_vectors)

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






