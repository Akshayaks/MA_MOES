import numpy as np
import os


if __name__ == "__main__":
    sphere_indv_erg = np.load("./BB_opt_indv_erg_random_maps_4_agents_sphere.npy",allow_pickle=True)
    sphere_indv_erg = sphere_indv_erg.ravel()[0]

    avg_indv_erg = np.load("./results_canebrake/BB_opt_indv_erg_4_agents.npy",allow_pickle=True)
    avg_indv_erg = avg_indv_erg.ravel()[0]
    breakpoint()

    for file in os.listdir("./build_prob/random_maps/"):
        if file not in avg_indv_erg.keys():
            print("Not in BB")
            continue
        total_file = "build_prob/random_maps/"+file
        print(total_file)
        if  total_file not in sphere_indv_erg.keys():
            print("Not in sphere")
            continue

        indv_wt = avg_indv_erg[file]
        indv_sp = sphere_indv_erg[total_file]
        
        print("indv erg wt: ", indv_wt)
        print("indv erg sph: ", indv_sp)
        print("minmax wt: ", max(indv_wt))
        print("minmax sp: ", max(indv_sp))
        breakpoint()
