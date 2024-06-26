import networkx as nx

from dataloader.data import Data

class Test(Data):
	def __init__(self, path, args):
		Data.__init__(self, path, args)
		# self.url = None#已经准备好，不用下载
		# self.fname = 'test.zip'
		# self.ftype = 'zip'
		self.has_class = 0#True
		self.raw_edgelist_fname = self.rawdir / 'test.txt'
		self.raw_label_fname = self.rawdir / 'test.label'


	def download_data(self):
		print('already have data...')
		pass

	def build_graph(self):
		self.G = nx.read_edgelist(self.raw_edgelist_fname,nodetype=int)
		return