import numpy as np
import common
from utils import *
from gen_maps import map_from_start_pos
from ergodic_coverage_single_integrator import ErgCover
import jax.numpy as jnp
import ergodic_metric_single_integrator as EM

if __name__ == "__main__":
    s1 = np.array([0.2,0.4,0.0])
    s2 = np.array([0.8,0.8,0.0])

    map_from_start_pos(s1,"./build_prob/s1.pickle")
    map_from_start_pos(s2,"./build_prob/s2.pickle")

    # file = "./build_prob/random_maps/random_map_0.pickle"
    # pbm = common.LoadProblem(file, 1, pdf_list=True)
    
    # file = "./build_prob/random_maps/random_map_1.pickle"
    # pbm = common.LoadProblem(file, 1, pdf_list=True)
    
    file = "./build_prob/unimodal_0.pickle"
    pbm = common.LoadProblem(file, 1, pdf_list=True)

    file_s1 = "./build_prob/s1.pickle"
    file_s2 = "./build_prob/s2.pickle"
    pbm_s1 = common.LoadProblem(file_s1, 1, pdf_list=True)
    pbm_s2 = common.LoadProblem(file_s2, 1, pdf_list=True)

    display_map_simple(pbm,np.concatenate((s1,s2)))
    display_map_simple(pbm_s1,s1)
    display_map_simple(pbm_s2,s2)

    EC = EM.ErgCalc(pbm.pdfs[0].flatten(),1,100,10,100)
    phik = EC.phik*np.sqrt(EC.lamk)
    
    EC = EM.ErgCalc(pbm_s1.pdfs[0].flatten(),1,100,10,100)
    phik_s1 = EC.phik*np.sqrt(EC.lamk)

    EC = EM.ErgCalc(pbm_s2.pdfs[0].flatten(),1,100,10,100)
    phik_s2 = EC.phik*np.sqrt(EC.lamk)

    print("L2 norm between map and s1: ", np.linalg.norm(phik-phik_s1))
    print("L2 norm between map and s2: ", np.linalg.norm(phik-phik_s2))

    pdf = jnp.asarray(pbm.pdfs[0].flatten())
    control, erg, iters = ErgCover(pdf, 1, pbm.nA, np.array([0.2,0.4]), 10, pbm.pix, 1000, False, None, stop_eps=-1,grad_criterion=True)
    print("Ergodicity achieved: ", erg[-1])
    _, tj = EM.GetTrajXY(control, np.array([0.2,0.4]))
    # print("Trajectory: ", tj)
    display_map(pbm,s1,{0:[0],1:[1]},tj=[tj,tj])

    control, erg, iters = ErgCover(pdf, 1, pbm.nA, np.array([0.8,0.8]), 10, pbm.pix, 1000, False, None, stop_eps=-1,grad_criterion=True)
    print("Ergodicity achieved: ", erg[-1])
    _, tj = EM.GetTrajXY(control, np.array([0.8,0.8]))
    # print("Trajectory: ", tj)
    display_map(pbm,s2,{0:[0],1:[1]},tj=[tj,tj])



