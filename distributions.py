import numpy as np
from scipy.stats import multivariate_normal as mv
from PIL import Image, ImageDraw


def gauMixDistrib(num_nodes=2, num_pts=100):
	"""
	borrowed from Ian's rt_ergodic_control package.
	"""

	num_pts = num_pts
	g = np.meshgrid(*[np.linspace(0, 1, num_pts) for _ in range(2)])
	grid = np.c_[g[0].ravel(), g[1].ravel()]

	means = [np.random.uniform(0.2, 0.8, size=(2,))
	                    for _ in range(num_nodes)]

	variances  = [np.random.uniform(0.05, 0.2, size=(2,))**2
	                    for _ in range(num_nodes)]

	val = np.zeros(grid.shape[0])
	for m, v in zip(means, variances):
		innerds = np.sum((grid-m)**2 / v, 1)
		val += np.exp(-innerds/2.0)# / np.sqrt((2*np.pi)**2 * np.prod(v))
	# normalizes the distribution
	val /= np.sum(val)
	return val.reshape((num_pts, num_pts))


def gaussianMixtureDistribution(n_gaus, npix, w=None, mus=None, covs=None):
	if w is None:
		# w = np.random.random(n_gaus)
		w = np.ones(n_gaus)/n_gaus
	if mus is None:
		mus = np.zeros((n_gaus, 2))
		mus[:, 0] = np.random.uniform(.1, .9, n_gaus) # [0.1, 0.9]
		mus[:, 1] = np.random.uniform(.1, .9, n_gaus)
	if covs is None:
		covs = np.zeros((n_gaus, 2, 2))
		covs[:, 0, 0] = (np.random.uniform(.05, .5, n_gaus)) ** 2 # [0.1,0.1]
		covs[:, 1, 1] = (np.random.uniform(.05, .5, n_gaus)) ** 2
		corr = np.random.uniform(-.9, .9, n_gaus)
		corr = np.random.uniform(0, 0, n_gaus)
		cov_diag = np.sqrt(covs[:, 0, 0]) * np.sqrt(covs[:, 1, 1]) * corr
		covs[:, 0, 1] = cov_diag
		covs[:, 1, 0] = cov_diag

	assert w.shape == (n_gaus,), "incorrect shape for w"
	assert mus.shape == (n_gaus, 2), "incorrect shape for mus"
	assert covs.shape == (n_gaus, 2, 2), "incorrect shape for covs"

	w = w / np.sum(w)

	xg, yg = np.meshgrid(
		np.linspace(0, 1, npix), np.linspace(0, 1, npix), indexing='ij'
	)
	flatgrid = np.stack([xg.flatten(), yg.flatten()], axis=1)

	p = np.zeros((npix, npix))
	for i in range(n_gaus):
		p += w[i]*mv.pdf(flatgrid, mean=mus[i, :], cov=covs[i, :, :]).reshape((npix, npix))

	return p / np.sum(p)


def roadwayDistribution(n_road, n_pix, width=None, v_onroad=1, v_offroad=0):
	if width is None:
		width = n_pix // 20

	start_side = np.random.randint(0, 3, n_road)
	start = np.random.uniform(0, 1, n_road)
	end_side = (start_side + 1 + np.random.randint(0, 2, n_road)) % 4
	end = np.random.uniform(0, 1, n_road)

	start_px = np.zeros(n_road)
	idx = (start_side == 0) | (start_side == 2)
	start_px[idx] = np.round(start[idx] * (n_pix - 1)).astype(np.int32)
	start_px[start_side == 1] = n_pix - 1
	start_py = np.zeros(n_road)
	start_py[~idx] = np.round(start[~idx] * (n_pix - 1)).astype(np.int32)
	start_py[start_side == 2] = n_pix - 1

	end_px = np.zeros(n_road)
	idx = (end_side == 0) | (end_side == 2)
	end_px[idx] = np.round(end[idx] * (n_pix - 1)).astype(np.int32)
	end_px[end_side == 1] = n_pix - 1
	end_py = np.zeros(n_road)
	end_py[~idx] = np.round(end[~idx] * (n_pix - 1)).astype(np.int32)
	end_py[end_side == 2] = n_pix - 1

	im = Image.new('F', (n_pix, n_pix))
	draw = ImageDraw.Draw(im)
	for i in range(n_road):
		draw.line((start_px[i], start_py[i], end_px[i], end_py[i]), 1, width=width)
	p = np.array(im)
	return p / np.sum(p)

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	n = 10
	p = gaussianMixtureDistribution(n, 100, w=np.ones(n))
	plt.contour(p)
	plt.show()

	p = roadwayDistribution(n, 100)
	plt.contour(p)
	plt.show()