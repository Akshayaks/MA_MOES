import os
import common
import numpy as np

if __name__ == "__main__":
    cnt = np.zeros(12)
    c = 0
    for file in os.listdir("build_prob/random_maps/"):
        pbm_file = "build_prob/random_maps/"+file
        print("\nFile: ", pbm_file)
        problem = common.LoadProblem(pbm_file, 4, pdf_list=True)
        cnt[len(problem.pdfs)] += 1
        c += 1
    
    for i in range(12):
        print("\ni: ", i, cnt[i])
    print("Total: ", c)
