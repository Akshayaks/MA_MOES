from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
import numpy as np
from more_itertools import set_partitions
import itertools
import math

np.random.seed(100)

def gen_start_pos(folder, n_agents):
  #Generate random starting positions (x,y) for the agents
  s0_arr = {}
  for pbm_file in os.listdir(folder):
    pos = np.random.uniform(0,1,2*n_agents)

    s0 = []
    k = 0
    for i in range(n_agents):
      s0.append(pos[k])
      s0.append(pos[k+1])
      s0.append(0)        #Starting orientation is always 0
      k += 2
    s0_arr[pbm_file] = np.array(s0)
    np.save("start_pos.npy",s0_arr)
  return s0_arr

def random_start_pos(n_agents):
  #Generate random starting positions (x,y) for the agents
  pos = np.random.uniform(0,1,2*n_agents)

  s0 = []
  k = 0
  for i in range(n_agents):
    s0.append(pos[k])
    s0.append(pos[k+1])
    s0.append(0)        #Starting orientation is always 0
    k += 2

  return np.array(s0)

#Add lines to add the case when all the maps are allotted to one agent
'''
Generates all the allovation combinations for given n_a and n_o
'''
def generate_allocations(n_obj,n_agents):
  objs = np.arange(0,n_obj)
  comb = list(set_partitions(objs, n_agents))
  alloc_comb = []
  for c in comb:
    alloc_comb.append(list(itertools.permutations(c)))
  alloc_comb = list(itertools.chain.from_iterable(alloc_comb))
  return alloc_comb


'''
Display function that can take any number of maps and plot the trajectory if given
'''
def display_map(pbm,start_pos,pbm_file=None,tj=None,window=None,r=None,title=None):
  x = np.linspace(0,100,num=100)
  y = np.linspace(0,100,num=100)
  X,Y = np.meshgrid(x,y)

  n_obj = len(pbm.pdfs)
  n_col = math.ceil(n_obj/2)
  if n_col == 1:
    n_col += 1
  n_agents = int(len(start_pos)/3)
  print(n_agents)

  fig, axs = plt.subplots(2, n_col,figsize=(5,5))
  l = 0
  colors = ["green","red","yellow","blue","cyan","magento","#eeefff","#ffa500","#a020f0","#ffc0cb"]

  for i in range(2):
    for j in range(n_col):
      print(l)
      axs[i,j].contourf(X, Y, pbm.pdfs[l], levels=np.linspace(np.min(pbm.pdfs[l]), np.max(pbm.pdfs[l]),100), cmap='gray')
      axs[i,j].set_title("Info Map "+str(l+1))
      for k in range(n_agents):
        axs[i,j].plot(pbm.s0[k*3]*100,pbm.s0[k*3+1]*100, marker="o", markersize=5, markerfacecolor=colors[k], markeredgecolor=colors[k])
        if window:
          w_size = window[l,k]
          x0 = pbm.s0[k*3]*100
          y0 = pbm.s0[k*3+1]*100
          h1 = max(0,y0-w_size)
          h2 = min(100,y0+w_size)
          w1 = max(0,x0-w_size)
          w2 = min(100,x0+w_size)
          axs[i,j].add_patch(Rectangle((w1,h1),
                           w2-w1, h2-h1,
                           fc ='none', 
                           ec ='w',
                           lw = 1) )
        if tj != None:
          axs[i,j].plot(tj[:,0]*100,tj[:,1]*100)
      l += 1
      if l == n_obj:
        break
    if l == n_obj:
      break
  if title:
    fig.suptitle(title)
  if pbm_file:
    plt.savefig("./test_cases_win_results/"+pbm_file+".jpg")
  plt.show()
  return

'''
Display the map and the cropped portion of the map (window approach)
'''
def display_both(pdf,s0,h1,h2,w1,w2,pdf_zeroed,start_pos,tj,w_size):
   h,w = pdf.shape
   print(pdf.shape)
   x = np.linspace(0,w,num=w)
   y = np.linspace(0,h,num=h)
   X,Y = np.meshgrid(x,y)

   fig, axs = plt.subplots(2, 2)
   axs[0,0].contourf(X, Y, pdf, levels=np.linspace(np.min(pdf), np.max(pdf),100), cmap='gray')
   axs[0,0].set_title('Info Map')

   axs[0,0].plot(s0[0]*100,s0[1]*100,"go--")
   axs[0,0].add_patch(Rectangle((w1,h1),
                           w2-w1, h2-h1,
                           fc ='none', 
                           ec ='w',
                           lw = 1) )
   x1 = np.linspace(0,w2-w1,num=w2-w1)
   y1 = np.linspace(0,h2-h1,num=h2-h1)
   X1,Y1 = np.meshgrid(x1,y1)

   axs[0,1].contourf(X1, Y1, pdf_zeroed, levels=np.linspace(np.min(pdf), np.max(pdf),100), cmap='gray')
   axs[0,1].plot(start_pos[0]*(w2-w1),start_pos[1]*(h2-h1),"go--")
   axs[0,1].plot(tj[:,0]*(w2-w1),tj[:,1]*(h2-h1))
   # plt.pause(1)
   # plt.savefig("./windows/"+str(w_size)+"_"+str(start_pos[0]*100)+"_"+str(start_pos[1]*100)+".jpg")
   plt.show()


