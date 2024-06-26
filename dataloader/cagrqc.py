import networkx as nx

from dataloader.data import Data

class caGrQc(Data):
	def __init__(self, path, args):
		Data.__init__(self, path, args)
		self.url = 'https://snap.stanford.edu/data/ca-GrQc.txt.gz'
		self.fname = 'ca-GrQc.txt.gz'
		self.ftype = 'gz'

	def build_graph(self):
		raw_edgelist_fname = self.rawdir / "ca-GrQc.txt"
		self.G = nx.read_edgelist(raw_edgelist_fname,nodetype=int)
		return