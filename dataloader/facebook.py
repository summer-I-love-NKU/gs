import networkx as nx

from dataloader.data import Data

class Facebook(Data):
	def __init__(self, path, args):
		Data.__init__(self, path, args)
		self.url = 'http://snap.stanford.edu/data/facebook_large.zip'
		self.fname = 'facebook_large.zip'
		self.ftype = 'zip'

	def build_graph(self):
		raw_edgelist_fname = self.rawdir / 'facebook_large' / 'musae_facebook_edges.csv'
		self.G = nx.read_edgelist(raw_edgelist_fname.as_posix(),delimiter=',',nodetype=int)
		return