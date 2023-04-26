from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
import numpy as np
from more_itertools import set_partitions
import itertools
import math
import os
import cProfile
import re

np.random.seed(100)

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def gen_start_pos(folder, n_agents):
  #Generate random starting positions (x,y) for the agents
  s0_arr = {}
  for pbm_file in os.listdir(folder):
    pos = np.random.uniform(0,1,2*n_agents)
    theta = np.random.uniform(0,2*np.pi,n_agents)

    s0 = []
    k = 0
    for i in range(n_agents):
      s0.append(pos[k])
      s0.append(pos[k+1])
      s0.append(theta[i])        #Starting orientation is always 0
      k += 2
    s0_arr[pbm_file] = np.array(s0)
    np.save("start_pos_ang_random_10_agents.npy",s0_arr)
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
  # print("num comb: ", len(comb))
  # print("comb: ", comb)
  alloc_comb = []
  for c in comb:
    alloc_comb.append(list(itertools.permutations(c)))
  alloc_comb = list(itertools.chain.from_iterable(alloc_comb))
  return alloc_comb


'''
Display function that can take any number of maps and plot the trajectory if given
'''
def display_map_results(pbm,start_pos,pbm_file=None,tj=None,window=None,r=None,title=None):
  x = np.linspace(0,100,num=100)
  y = np.linspace(0,100,num=100)
  X,Y = np.meshgrid(x,y)

  pbm.s0 = start_pos

  n_obj = len(pbm.pdfs)
  n_col = math.ceil(n_obj/2)
  if n_col == 1:
    n_col += 1
  n_agents = int(len(start_pos)/3)
  print("Number of agents: ", n_agents)

  fig, axs = plt.subplots(2, n_col,figsize=(5,5))
  l = 0
  color = ["green","red","yellow","blue","cyan","#ee00ff","#ffa500","#a020f0","#ffc0cb","#007ccc"]
  colors = ["red", "blue", "green"]

  for i in range(2):
    for j in range(n_col):
      print(l)
      axs[i,j].contourf(X, Y, pbm.pdfs[l], levels=np.linspace(np.min(pbm.pdfs[l]), np.max(pbm.pdfs[l]),100), cmap='gray')
      axs[i,j].set_title("Info Map "+str(l+1))
      for k in range(n_agents):
        # axs[i,j].plot(pbm.s0[k*3]*100,pbm.s0[k*3+1]*100, marker="o", markersize=5, markerfacecolor=colors[k], markeredgecolor=colors[k])
        # if l == 1:
        #   axs[i,j].plot(pbm.s0[2*3]*100,pbm.s0[2*3+1]*100, marker="o", markersize=7, markerfacecolor=colors[2], markeredgecolor=colors[2])
        # elif l == 2:
        #   axs[i,j].plot(pbm.s0[0*3]*100,pbm.s0[0*3+1]*100, marker="o", markersize=7, markerfacecolor=colors[0], markeredgecolor=colors[0])
        # elif l == 0:
        #   axs[i,j].plot(pbm.s0[1*3]*100,pbm.s0[1*3+1]*100, marker="o", markersize=7, markerfacecolor=colors[1], markeredgecolor=colors[1])
        # elif l == 3:
        #   axs[i,j].plot(pbm.s0[2*3]*100,pbm.s0[2*3+1]*100, marker="o", markersize=7, markerfacecolor=colors[2], markeredgecolor=colors[2])
        # elif l == 4:
        #   axs[i,j].plot(pbm.s0[0*3]*100,pbm.s0[0*3+1]*100, marker="o", markersize=7, markerfacecolor=colors[0], markeredgecolor=colors[0])
        # axs[i,j].plot(pbm.s0[l*3]*100,pbm.s0[l*3+1]*100, marker="o", markersize=5, markerfacecolor=colors[l], markeredgecolor=colors[l])
        axs[i,j].plot(pbm.s0[k*3]*100,pbm.s0[k*3+1]*100, marker="o", markersize=5, markerfacecolor=color[k], markeredgecolor=color[k])
        if window:
          w_size = 15 #window[l,k]
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
          if l == 0:
            axs[i,j].plot(tj[l][:,0]*100,tj[l][:,1]*100,color="red",linewidth=2) #green
          elif l == 1:
            axs[i,j].plot(tj[l][:,0]*100,tj[l][:,1]*100,color="blue",linewidth=2) #red
          elif l == 2:
            axs[i,j].plot(tj[l][:,0]*100,tj[l][:,1]*100,color="green",linewidth=2) #blue
          elif l == 3:
            axs[i,j].plot(tj[l][:,0]*100,tj[l][:,1]*100,color="green",linewidth=2)
          else:
            axs[i,j].plot(tj[l][:,0]*100,tj[l][:,1]*100,color="red",linewidth=2)
      l += 1
      if l == n_obj:
        break
    if l == n_obj:
      break
  if title:
    fig.suptitle(title)
  if pbm_file:
    plt.savefig("./random_maps/bb_"+pbm_file+".jpg")
  plt.show()
  return

def display_map(pbm,start_pos,pbm_file=None,tj=None,title=None,ref=None):
    x = np.linspace(0,100,num=100)
    y = np.linspace(0,100,num=100)
    X,Y = np.meshgrid(x,y)

    # pbm.s0 = start_pos

    n_obj = len(pbm.pdfs)
    n_col = math.ceil(n_obj/2)
    if n_col == 1:
      n_col += 1
    n_agents = int(len(start_pos)/3)
    n_agents = 1
    print("Number of agents: ", n_agents)

    fig, axs = plt.subplots(2, n_col,figsize=(5,5))
    l = 0
    color = ["green","red","yellow","blue","cyan","#ee00ff","#ffa500","#a020f0","#ffc0cb","#007ccc"]
    colors = ["red", "blue", "green", "yellow"]

    for i in range(2):
      for j in range(n_col):
        print(l)
        axs[i,j].contourf(X, Y, pbm.pdfs[l], levels=np.linspace(np.min(pbm.pdfs[l]), np.max(pbm.pdfs[l]),100), cmap='gray')
        axs[i,j].set_title("Info Map "+str(l+1))
        for k in range(n_agents):
          axs[i,j].plot(pbm.s0[k*3]*100,pbm.s0[k*3+1]*100, marker="o", markersize=5, markerfacecolor=colors[k], markeredgecolor=colors[k])
          if tj != None:
            if l == 0:
              breakpoint()
              axs[i,j].plot(tj[l][:,0]*100,tj[l][:,1]*100,color="red",linewidth=2) #green
              if ref:
                axs[i,j].plot(ref[l][:,0]*100,ref[l][:,1]*100,color="green",linewidth=2) #green
            elif l == 1:
              axs[i,j].plot(tj[l][:,0]*100,tj[l][:,1]*100,color="blue",linewidth=2) #red
            elif l == 2:
              axs[i,j].plot(tj[l][:,0]*100,tj[l][:,1]*100,color="green",linewidth=2) #blue
            elif l == 3:
              axs[i,j].plot(tj[l][:,0]*100,tj[l][:,1]*100,color="green",linewidth=2)
            else:
              axs[i,j].plot(tj[l][:,0]*100,tj[l][:,1]*100,color="red",linewidth=2)
        l += 1
        if l == n_obj:
          break
      if l == n_obj:
        break
    if title:
      fig.suptitle(title)
    if pbm_file:
      plt.savefig("./random_maps/bb_"+pbm_file+".jpg")
    plt.legend(["Reference trajectory", "Actual trajectory"])
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


# cProfile.run("random_start_pos(5)")
# a = generate_allocations(12,4)
# print("Num: ", len(a))
# print("a: ", a)

gen_start_pos("./build_prob/random_maps_20/",10)
