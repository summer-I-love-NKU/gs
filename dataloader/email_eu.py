import networkx as nx

from dataloader.data import Data

class Email_EU(Data):
	def __init__(self, path, args):
		Data.__init__(self, path, args)
		self.url = ['https://snap.stanford.edu/data/email-Eu-core.txt.gz','https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz']
		self.fname = ['email-Eu-core.txt.gz','email-Eu-core-department-labels.txt.gz']
		self.ftype = 'gz' #多个文件必须类型相同
		self.has_class=True
		self.raw_edgelist_fname = self.rawdir / 'email-Eu-core.txt'
		self.raw_label_fname = self.rawdir / 'email-Eu-core-department-labels.txt'

	def build_graph(self):
		self.G = nx.read_edgelist(self.raw_edgelist_fname, nodetype=int)
		return