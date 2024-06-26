import numpy as np
import networkx as nx

from dataloader.data import Data

class SBM(Data):
	def __init__(self, path, args):
		Data.__init__(self, path, args)
		self.seed = args.dataset_seed
		self.n_cliques = args.n_cliques
		self.clique_size = args.clique_size
		self.tries = args.sbm_tries
		self.intra_prob = args.sbm_intra_prob
		self.inter_prob = args.sbm_inter_prob

	def download_data(self):
		pass

	def build_graph(self):
		for _ in range(self.tries):
			sizes = [self.clique_size]*self.n_cliques
			probs = np.full((self.n_cliques,self.n_cliques), self.inter_prob)
			np.fill_diagonal(probs, self.intra_prob)
			H = nx.stochastic_block_model(sizes, probs, seed=self.seed)
			if nx.is_connected(H):
				break
		self.G = H
		return