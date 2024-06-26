import networkx as nx

from dataloader.data import Data

class Youtube(Data):
	def __init__(self, path, args):
		Data.__init__(self, path, args)
		self.url = 'http://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz'
		self.fname = 'com-youtube.ungraph.txt.gz'
		self.ftype = 'gz'

	def build_graph(self):
		raw_edgelist_fname = self.rawdir / 'com-youtube.ungraph.txt'
		self.G = nx.read_edgelist(raw_edgelist_fname,nodetype=int)
		return