import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import common
import cv2
from matplotlib.cbook import get_sample_data
from utils import *

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
file = "./build_prob/random_maps/random_map_28.pickle"
pbm = common.LoadProblem(file, n_agents=4, pdf_list=True)

X = np.arange(0, 100, 1)
Y = np.arange(0, 100, 1)
X, Y = np.meshgrid(X, Y)
Z1 = pbm.pdfs[0]*1000
Z2 = pbm.pdfs[1]*1000
Z3 = pbm.pdfs[2]*1000
Z4 = pbm.pdfs[3]*1000
Z5 = pbm.pdfs[4]*1000
Z = np.ones((100,100))

# Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.7)
# surf = ax.plot_surface(X, Y, Z2+2, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.7)
# surf = ax.plot_surface(X, Y, Z3+3, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.7)
# surf = ax.plot_surface(X, Y, Z4+10, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.7)
# surf = ax.plot_surface(X, Y, Z5+12, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.7)

fn = get_sample_data("/home/akshaya/Downloads/rosbots_in_an_area.png", asfileobj=False)

# img = cv2.imread(fn)
# Change the color from BGR to RGB
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# img = img[:100,:100]

img = np.zeros((100,100))

# Orgird to store data
x, y = np.ogrid[0:img.shape[0], 0:img.shape[1]]
# In Python3 matplotlib assumes rgbdata in range 0.0 to 1.0
img = img / 255
# arr = read_png(fn)
# # 10 is equal length of x and y axises of your surface
# stepX, stepY = 101. / img.shape[0], 101. / img.shape[1]

# X1 = np.arange(0, 100, stepX)
# Y1 = np.arange(0, 100, stepY)
# X1, Y1 = np.meshgrid(X1, Y1)
# stride args allows to determine image quality 
# stride = 1 work slow
# fig = plt.figure()
# gca do not work thus use figure objects inbuilt function.
# ax = fig.add_subplot(projection='3d')

# Plot data
s2 = ax.plot_surface(X,Y, img, cmap=cm.YlOrRd, linewidth=0, antialiased=False, alpha=0.7) #np.atleast_2d(0), rstride=10, cstride=10, facecolors=img)

# fig.savefig("results.png")
# fig.show()
# ax.plot_surface(x, y, 2.01, rstride=1, cstride=1, facecolors=img)

# Customize the z axis.
ax.set_zlim(0, 3.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
# ax.set_frame_on(False)
# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# ax.zaxis.set_ticklabels([])

# for line in ax.xaxis.get_ticklines():
#     line.set_visible(False)
# for line in ax.yaxis.get_ticklines():
#     line.set_visible(False)
# for line in ax.zaxis.get_ticklines():
#     line.set_visible(False)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
# fig.colorbar(s2, shrink=0.5, aspect=5)

display_map(pbm,np.array([0.1,0.1,0,0.6,0.7,0]),{0:[0],1:[1]})

plt.show()