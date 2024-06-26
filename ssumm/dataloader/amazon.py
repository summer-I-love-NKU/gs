import networkx as nx

from dataloader.data import Data

class Amazon(Data):
	def __init__(self, path, args):
		Data.__init__(self, path, args)
		self.url = 'http://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz'
		self.fname = 'com-amazon.ungraph.txt.gz'
		self.ftype = 'gz'

	def build_graph(self):
		raw_edgelist_fname = self.rawdir / 'com-amazon.ungraph.txt.gz'
		self.G = nx.read_edgelist(raw_edgelist_fname,nodetype=int)
		return