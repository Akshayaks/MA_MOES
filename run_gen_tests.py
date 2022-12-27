import jax.numpy as jnp
import numpy as np 
import matplotlib.pyplot as plt
from ergodic_coverage import ErgCover
import scalarize
import common as cm


def main_MOES_Fourier():
  nA = 100
  n_fourier = 5
  pbm_file = "build_prob/instances/MOES_example_rosbot.pickle"
  cm.GenMOESProblemFourier(nA,n_fourier, pbm_file)

def main_MOESO2Simple(): # two-obj
  nA = 100
  pbm_file = "build_prob/instances/simple_3_peaks_map.pickle"
  # # ############## Gen tests #############
  cm.GenMOESProblemO2Simple(nA, pbm_file)

def main_MOESO3Simple(): # three-obj
  nA = 100
  pbm_file = "build/instances/MOES-O3-peaks_pix_100_multimodal_3.pickle"
  # # ############## Gen tests #############
  cm.GenMOESProblemO3Simple(nA, pbm_file)

def main_MOESO2Random():
  nA = 100
  nGau = 2
  k=0
  pbm_file = "build/instances/MOES-O2-nA_"+str(nA)+"_nGau_"+str(nGau)+"_pix_100_random_k_"+str(k)+".pickle"
  # # ############## Gen tests #############
  cm.GenMOESProblemO2Random(nGau, k, nA, pbm_file)

if __name__ == "__main__":
  main_MOES_Fourier()
  
  