from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

def display_map(pbm,start_pos,alloc,pbm_file=None,tj=None,title=None,ref=None,collision_points=None):
    x = np.linspace(0,100,num=100)
    y = np.linspace(0,100,num=100)
    X,Y = np.meshgrid(x,y)

    pbm.s0 = start_pos

    n_obj = len(pbm.pdfs)
    n_col = math.ceil(n_obj/2)
    if n_col == 1:
      n_col += 1
    n_agents = int(len(start_pos)/3)

    fig, axs = plt.subplots(2, n_col,figsize=(5,5))
    l = 0
    colors = ["red", "blue", "green", "yellow"] #Colors for each agent

    for i in range(2):
      for j in range(n_col):
        print(l)
        axs[i,j].contourf(X, Y, pbm.pdfs[l], levels=np.linspace(np.min(pbm.pdfs[l]), np.max(pbm.pdfs[l]),100), cmap='gray')
        axs[i,j].set_title("Info Map "+str(l+1))

        for k in range(n_agents):
          if l in alloc[k]:
            axs[i,j].plot(pbm.s0[k*3]*100,pbm.s0[k*3+1]*100, marker="o", markersize=5, markerfacecolor=colors[k], markeredgecolor=colors[k])
            if tj != None:
              axs[i,j].plot(tj[l][:,0]*100,tj[l][:,1]*100,color=colors[k],linewidth=2)
            break
        l += 1
        if l == n_obj:
          break
      if l == n_obj:
        break
    if title:
      fig.suptitle(title)
    if pbm_file:
      plt.savefig("./random_maps/bb_"+pbm_file+".jpg")
    if ref:
      plt.legend(["Reference trajectory", "Actual trajectory"])
    plt.show()

    
    if collision_points != None:
      for c in range(len(collision_points)):
        print(collision_points[c])
        for k in range(n_agents):
          plt.plot(tj[k][:collision_points[c][0],0]*100,tj[k][:collision_points[c][0],1]*100,color=colors[k],linewidth=2)
        c_x = [collision_points[c][1]*100]
        c_y = [collision_points[c][2]*100]
        plt.scatter(c_x,c_y,marker='X',linewidths=1.5)
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

# def animate_traj(trajectories):
#   print("Animating the plot")
#   t = np.arange(0,len(trajectories[0]))
#   n = len(trajectories)

#   t = np.linspace(0, t_flight, 100)
#   x = u*np.cos(theta)*t
#   y = u*np.sin(theta)*t - 0.5*g*t**2

#   fig, ax = plt.subplots()
#   line, = ax.plot(x, y, color='k')

#   xmin = x[0]
#   ymin = y[0]
#   xmax = max(x)
#   ymax = max(y)
#   xysmall = min(xmax,ymax)
#   maxscale = max(xmax,ymax)
#   circle = plt.Circle((xmin, ymin), radius=np.sqrt(xysmall))
#   ax.add_patch(circle)

#   def update(num, x, y, line, circle):
#       line.set_data(x[:num], y[:num])
#       circle.center = x[num],y[num]
#       line.axes.axis([0, max(np.append(x,y)), 0, max(np.append(x,y))])

#       return line,circle

#   ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, line, circle],
#                                 interval=25, blit=True)

#   ani.save('projectile.gif')
#   plt.show()

# cProfile.run("random_start_pos(5)")
# a = generate_allocations(12,4)
# print("Num: ", len(a))
# print("a: ", a)

# gen_start_pos("./build_prob/random_maps_20/",10)
