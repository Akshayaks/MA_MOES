from utils import *
import numpy as np
import common
import os
import pdb
from similarity_between_maps import k_means_clustering 

n_agents = 2
start_pos_file = "./start_pos_random_2_agents.npy"
start_pos = np.load(start_pos_file,allow_pickle=True)

for file in os.listdir("./build_prob/random_maps/"):
    if file != "random_map_28.pickle":
        continue
    pbm_file_complete = "./build_prob/random_maps/" + file
    pbm = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)
    if len(pbm.pdfs) != 5:
        continue
    print("Name: ", file)
    s = start_pos.item().get(file)
    print("start pos: ", s)
    print("\nNumber of maps: ", len(pbm.pdfs))
    clusters = k_means_clustering(pbm.pdfs,n_agents)
    print("\nClusters: ", clusters)

    display_map(pbm,s,pbm_file=None,tj=None,window=None,r=None,title=None)


