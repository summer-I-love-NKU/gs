import os
import subprocess
import numpy as np
import networkx as nx
from pathlib import Path
from argparse import Namespace
from torch_geometric.data import download_url, extract_tar, extract_gz, extract_zip

class Data:
	def __init__(self, path: str, args: Namespace):
		self.path = path
		self.name = args.dataset_name
		self.param_str, self.rawdir, self.procdir = self.create_data_dirs(args)
		self.rawdir.mkdir(parents=True, exist_ok=True)
		self.procdir.mkdir(parents=True, exist_ok=True)
		self.edgelist_fname = (self.procdir / (self.param_str + '.edgelist')).as_posix()
		self.label_fname = (self.procdir / (self.param_str + '.label')).as_posix()
		self.node_class_dict = None
		self.node_class_list = None
		self.node_class_num = None
		return

	def preprocess(self):
		# 最大连通分量
		self.G = self.largest_connected_component(self.G);nodes1=list(self.G.nodes)
		self.G = self.rename_nodes(self.G)
		nx.write_edgelist(self.G, self.edgelist_fname, data=False)

		# 处理节点类别文件
		if hasattr(self, 'has_class') and self.has_class:
			nodes2 = list(self.G.nodes);
			d_node = dict(zip(nodes1, nodes2))
			new_labels={}
			labels=np.loadtxt(self.raw_label_fname,dtype=int,comments=None,delimiter=None).T
			labels=dict(zip(labels[0],labels[1]))
			d_class={};class_id=0#对于label也重新索引
			for k,v in labels.items():
				if v not in d_class:
					d_class[v]=class_id;class_id+=1
				if k not in d_node:
					continue
				new_labels [ d_node[k] ] = d_class[v]
			res=np.array([list(new_labels.keys()), list(new_labels.values())]).T
			self.node_class_dict=new_labels
			self.node_class_list = list(new_labels.values())
			self.node_class_num = max(self.node_class_list)+1
			np.savetxt(self.label_fname,res,"%d")
		return

	def largest_connected_component(self, H: nx.Graph):
		return H.subgraph(max(nx.connected_components(H), key=len))

	def rename_nodes(self, H: nx.Graph):
		# relabel nodes to integers indexed from 0 to n-1
		mapping = dict(zip(list(H.nodes()), range(len(H))))
		H = nx.relabel_nodes(H, mapping)
		return H

	def download_data(self):
		if self.url is None:
			os.makedirs(self.rawdir,exist_ok=True)
			return

		_ = subprocess.run(f"rm -rf {self.rawdir}", shell=True)

		print(f"Downloading {self.name} dataset...")
		if type(self.url) == str:
			download_url(f'{self.url}', self.rawdir)
		elif type(self.url) == list:
			for url in self.url:
				download_url(f'{url}', self.rawdir)

		assert self.ftype in ['tar', 'gz', 'zip']

		if type(self.fname) == str:
			exec(f'extract_{self.ftype}((self.rawdir / self.fname).as_posix(), self.rawdir)')
		elif type(self.fname) == list:
			for fname in self.fname:
				exec(f'extract_{self.ftype}((self.rawdir / fname).as_posix(), self.rawdir)')

		return

	def read_edgelist(self):
		return nx.read_edgelist(self.edgelist_fname,nodetype=int)#attn 节点类型！

	def num_nodes(self):
		return self.G.number_of_nodes()

	def num_edges(self):
		return self.G.number_of_edges()
	
	def get_adjacency_matrix(self):
		return nx.adjacency_matrix(self.G).astype(np.float32)
	
	def build_graph(self):
		# overload this to write your own dataset-specific function
		pass

	def create_data_dirs(self, args):
		if self.name == 'SBM':
			param_str = '_'.join([self.name]+[str(v) for v in [args.dataset_seed, args.n_cliques, args.clique_size, args.sbm_tries, args.sbm_intra_prob, args.sbm_inter_prob]])
			rawdir = Path(self.path) / param_str / 'raw'
			procdir = Path(self.path) / param_str / 'processed'
		else:
			param_str = self.name
			rawdir = Path(self.path) / 'raw'
			procdir = Path(self.path) / 'processed'
		return param_str, rawdir, procdir