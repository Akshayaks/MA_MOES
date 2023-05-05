import numpy as np
from miniball import miniball
import common
from ergodic_coverage import ErgCover
import ergodic_metric

def center_sphere(pdf_list,nA,pix):
    FC = []
    for pdf in pdf_list:
        EC = ergodic_metric.ErgCalc(pdf.flatten(),1,nA,10,pix)
        FC.append(EC.phik*np.sqrt(EC.lamk))
    # print("Length of each map feature vector: ", len(FC[0]))
    # FC[1] = FC[0]
    # print("Feature vector: ", FC)
    res = miniball(np.asarray(FC,dtype=np.double))
    pdf_FC = res["center"]
    # pdf_FC = np.divide(res["center"],np.sqrt(EC.lamk))
    minmax = res["radius"]
    # print("Sphere center: ", pdf_FC)
    print("Sphere radius: ", minmax)
    # print("Diff: ", pdf_FC - FC[0])
    # print("Diff2: ", pdf_FC - FC[1])
    breakpoint()
    pdf_recon = EC.get_recon(pdf_FC)
    return pdf_FC,pdf_recon

best_alloc_bb = np.load("./results_canebrake/BB_opt_Best_alloc_4_agents.npy",allow_pickle=True)
indv_erg_bb = np.load("./results_canebrake/BB_opt_indv_erg_4_agents.npy",allow_pickle=True)
best_alloc_bb = best_alloc_bb.ravel()[0]
indv_erg_bb = indv_erg_bb.ravel()[0]

n_agents = 4
n_scalar = 10

start_pos = np.load("./start_positions/start_pos_ang_random_4_agents.npy",allow_pickle=True)

best_alloc_bb = np.load("./BB_opt_best_alloc_random_maps_sparse_4_agents_sphere.npy",allow_pickle=True)
best_alloc_bb = best_alloc_bb.ravel()[0]

best_alloc_sim = np.load("./Results_npy_files/BB_similarity_clustering_random_maps_best_alloc_4_agents.npy",allow_pickle=True)
best_alloc_sim = best_alloc_sim.ravel()[0]

for file in best_alloc_bb.keys():
    small_alloc = False
    print("File: ", file)
    alloc = best_alloc_bb[file]
    print("Alloc: ", alloc)
    if alloc == []:
        continue
    # if file != "random_map_29.pickle":
    #     continue

    pbm_file_complete = file #"./build_prob/random_maps/" + file
    problem = common.LoadProblem(pbm_file_complete, n_agents, pdf_list=True)

    n_obj = len(problem.pdfs)
    problem.nA = 100

    problem.s0 = start_pos.item().get(file)
    pdf_list = problem.pdfs

    alloc_erg = {}
    alloc_erg_wt = {}

    #Find the upper bound for the alloc solution
    for k,v in alloc.items():
        print("alloc: ", v)
        if len(v) == 0:
            continue
        if len(v) > 1:
            pdf_phik,pdf = center_sphere([pdf_list[a] for a in v],problem.nA,problem.pix)
            print("The phik obtained from bounding sphere center: ", pdf_phik[:10])
            
            pdf_weighted = np.zeros((100,100))
            for vi in v:
                pdf_weighted += (1/len(v))*pdf_list[vi]
            pdf_weighted = np.asarray(pdf_weighted.flatten())
            EC = ergodic_metric.ErgCalc(pdf_weighted,1,problem.nA,n_scalar,problem.pix)
            print("The phik of the scalarized information map: ", EC.phik[:10])

            breakpoint()
            print("Diff between two pdfs: ", pdf.flatten()-pdf_weighted)

            breakpoint()

            pdf = np.asarray(pdf.flatten())
            # Just run ergodicity optimization for fixed iterations to get ergodic trajectory 
            control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True,direct_FC=pdf_phik)
            control_wt, erg, _ = ErgCover(pdf_weighted.flatten(), 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)

        else:
            # small_alloc = True
            # break
            pdf = pdf_list[v[0]]
            pdf = np.asarray(pdf.flatten())
            control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*k:3+3*k], n_scalar, problem.pix, 1000, False, None, grad_criterion=True)
            control_wt = control
        
        # Calculate individual ergodicities using the gotten trajectory
        for p in v:
            pdf_indv = np.asarray(pdf_list[p].flatten())
            EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
            alloc_erg[p] = EC.fourier_ergodic_loss(control,problem.s0[3*k:3+3*k])
            alloc_erg_wt[p] = EC.fourier_ergodic_loss(control_wt,problem.s0[3*k:3+3*k])
    if small_alloc:
        continue
    upper = max(alloc_erg)
    # print("Alloc: ", alloc)
    # print("Indv erg with minimal bounding sphere: ", alloc_erg)
    # print("Indv erg with wt map: ", alloc_erg_wt)
    # print("Incumbent allocation: ", alloc)
    # print("Incumber Ergodicities: ", alloc_erg)
    # print("Initial Upper: ", upper)

    # print("Individual erg: ", indv_erg_bb[file])
    # print("Minmax: ", max(indv_erg_bb[file]))

    print("Clusters from BB: ", best_alloc_bb[file])
    print("Clusters from minimal bounding spheres: ",)
    print("Clusters from similarity clustering: ", best_alloc_sim[file])
    breakpoint()


