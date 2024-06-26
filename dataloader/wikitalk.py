import networkx as nx

from dataloader.data import Data

class Wikitalk(Data):
	def __init__(self, path, args):
		Data.__init__(self, path, args)
		self.url = 'http://snap.stanford.edu/data/wiki-Talk.txt.gz'
		self.fname = 'wiki-Talk.txt.gz'
		self.ftype = 'gz'

	def build_graph(self):
		raw_edgelist_fname = self.rawdir / 'wiki-Talk.txt'
		self.G = nx.read_edgelist(raw_edgelist_fname,nodetype=int)
		return