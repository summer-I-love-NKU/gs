import numpy as np
import networkx as nx
import os.path as osp
from scipy.io import loadmat
from scipy.sparse.csgraph import connected_components

from dataloader.data import Data

class Blogcatalog(Data):
	def __init__(self, path, args):
		Data.__init__(self, path, args)
		self.url = 'http://leitang.net/code/social-dimension/data/blogcatalog.mat'
		self.fname = 'blogcatalog.mat'
		self.ftype = 'mat'

	def build_graph(self):
		# load adj
		adj = loadmat(osp.join(self.rawdir, self.fname))['network']
		adj.setdiag(0)
		adj.data = 1. * (adj.data > 0)
		adj[adj != 0] = 1

		# make the graph undirected
		adj = adj.maximum(adj.T)

		# select the largest connected component
		_, components = connected_components(adj)
		c_ids, c_counts = np.unique(components, return_counts=True)
		id_max_component = c_ids[c_counts.argmax()]
		select = components == id_max_component
		adj = adj[select][:, select]

		# remove self-loops
		adj = adj.tolil()
		adj.setdiag(0)
		adj = adj.tocsr()
		adj.eliminate_zeros()
		self.G = nx.from_scipy_sparse_array(adj)# nx.to_scipy_sparse_array ??
		assert nx.is_connected(self.G)

		return