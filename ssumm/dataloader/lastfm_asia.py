import networkx as nx

from dataloader.data import Data

class LastFM_Asia(Data):
	def __init__(self, path, args):
		Data.__init__(self, path, args)
		self.url = 'http://snap.stanford.edu/data/lastfm_asia.zip'
		self.fname = 'lastfm_asia.zip'
		self.ftype = 'zip'

	def build_graph(self):
		raw_edgelist_fname = self.rawdir / 'lasftm_asia' / 'lastfm_asia_edges.csv'
		self.G = nx.read_edgelist(raw_edgelist_fname.as_posix(),delimiter=',',nodetype=int)
		return