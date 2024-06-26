import networkx as nx

from dataloader.data import Data

class Email_Enron(Data):
	def __init__(self, path, args):
		Data.__init__(self, path, args)
		self.url = 'http://snap.stanford.edu/data/email-Enron.txt.gz'
		self.fname = 'email-Enron.txt.gz'
		self.ftype = 'gz'

	def build_graph(self):
		raw_edgelist_fname = self.rawdir / 'email-Enron.txt'
		self.G = nx.read_edgelist(raw_edgelist_fname,nodetype=int)
		return