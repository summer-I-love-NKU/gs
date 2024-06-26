import networkx as nx

from dataloader.data import Data

class Cora(Data):
	def __init__(self, path, args):
		Data.__init__(self, path, args)
		self.url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
		self.fname = 'cora.tgz'
		self.ftype = 'tar'

	def build_graph(self):
		raw_edgelist_fname = self.rawdir / "cora" / "cora.cites"
		self.G = nx.read_edgelist(raw_edgelist_fname,nodetype=int)
		return