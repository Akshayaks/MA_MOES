from explicit_allocation import *
import os
import pdb
import jax
import argparse
import common
import matplotlib.pyplot as plt

def generate_bands_graph(pbm_file,n_agents,n_scalar,nA):
	problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)

	n_scalar = 10
	n_obj = len(problem.pdfs)

	problem.nA = 100 
	nA = problem.nA
	
	start_pos = np.load("start_pos.npy",allow_pickle=True)

	problem.s0 = start_pos.item().get("4_maps_example_0.pickle")
	print("Read start position as: ",problem.s0)

	print("Agent start positions allotted!")

	alloc_comb = generate_allocations(n_obj,n_agents)
	print(alloc_comb)
	pdb.set_trace()

	erg_variation_alloc = {}

	erg_mat = np.zeros((len(alloc_comb),n_obj))   #For each allocation, we want the individual ergodicities

	pdf_list = problem.pdfs
	print("pdf_list: ", pdf_list[0].shape)
	trajectory = []

	for idx,alloc in enumerate(alloc_comb):
	  print(alloc)
	  erg_history = {}
	  for i in range(n_agents):
	    pdf = np.zeros((100,100))
	    if len(alloc[i]) > 1:
	      # for a in alloc[i]:
	        # pdf += (1/len(alloc[i]))*pdf_list[a]
	      pdf = scalarize_minmax([pdf_list[a] for a in alloc[i]],problem.s0[i*3:i*3+3],problem.nA)
	    else:
	      pdf = pdf_list[alloc[i][0]]
	    
	    pdf = jnp.asarray(pdf.flatten())
	    print("start_pos: ",problem.s0)
	    
	    #Just run ergodicity optimization for fixed iterations and see which agent achieves best ergodicity in that time
	    if len(alloc[i]) > 1:
	    	control, erg, _, erg1, erg2 = ErgCover(pdf, 1, problem.nA, problem.s0[3*i:3+3*i], n_scalar, problem.pix, 1000, False, None, grad_criterion=False,pdf1=pdf_list[alloc[i][0]].flatten(),pdf2=pdf_list[alloc[i][1]].flatten())
	    else:
	    	control, erg, _ = ErgCover(pdf, 1, problem.nA, problem.s0[3*i:3+3*i], n_scalar, problem.pix, 1000, False, None, grad_criterion=False)
	    
	    if len(alloc[i]) > 1:
	    	erg_history[alloc[i][0]] = erg1
	    	erg_history[alloc[i][1]] = erg2
	    else:
	    	erg_history[alloc[i][0]] = erg
	    for p in alloc[i]:
	      pdf_indv = jnp.asarray(pdf_list[p].flatten())
	      EC = ergodic_metric.ErgCalc(pdf_indv,1,problem.nA,n_scalar,problem.pix)
	      erg_mat[idx][p] = EC.fourier_ergodic_loss(control,problem.s0[3*i:3+3*i])
	  erg_variation_alloc[idx] = erg_history
	return erg_variation_alloc, alloc_comb


if __name__ == "__main__":
	pbm_file = "./build_prob/test_cases/4_maps_example_0.pickle"
	n_scalar = 10
	n_agents = 2
	nA = 100
	erg_variation, alloc_comb = generate_bands_graph(pbm_file,n_agents,n_scalar,nA)
	print("Shape of erg_variation: ", len(erg_variation))

	alloc_idx = 0
	n_obj = 4
	alloc_comb = [([1], [2, 3, 4]), ([2, 3, 4], [1]), ([1, 2], [3, 4]), ([3, 4], [1, 2]), ([2], [1, 3, 4]), ([1, 3, 4], [2]), ([1, 2, 3], [4]), ([4], [1, 2, 3]), ([2, 3], [1, 4]), ([1, 4], [2, 3]), ([1, 3], [2, 4]), ([2, 4], [1, 3]), ([3], [1, 2, 4]), ([1, 2, 4], [3])]
	# alloc_comb = [([1], [2, 3]), ([2, 3], [1]), ([1, 2], [3]), ([3], [1, 2]), ([2], [1, 3]), ([1, 3], [2])]

	for i in range(n_obj):
		for a in range(14): #For running through allocation combinations
			erg = erg_variation[a][i]
			iterations = np.arange(0,len(erg))
			plt.plot(iterations,erg,label=str(alloc_comb[a]))
		plt.title("Ergodicity variation over iterations on Map " + str(i+1))
		plt.xlabel("Iterations")
		plt.ylabel("Ergodicity")
		plt.legend(loc="upper right")
		plt.show()
		plt.pause(1)

