from collections import defaultdict
import networkx as nx
import numpy as np

class Args:
    pass

# check_input(file='Email_EU.edgelist')

#ok:    la  fb  sk  db

# 有问题： ca

class Data:

    def __init__(self, args):
        self.dataPath = args.dataPath
        self.dataset_name = args.dataPath.split('/')[-1].split('\\')[-1].split('.')[0]  # xxx.txt
        self.edgelist_fname = self.dataset_name + '.edgelist'
        self.has_label=False
        if hasattr(args, 'labelPath'):
            self.raw_label_fname = args.labelPath
            self.label_fname = self.dataset_name + '.label'
            self.has_label = True
            self.node_label_dict = None
            self.node_label_list = None
            self.node_label_num = None 
        return

    def preprocess(self):
        self.G=nx.read_edgelist(self.dataPath,nodetype=int)
        # 最大连通子图
        self.G = self.largest_connected_component(self.G)#nx.is_frozen(self.G) True
        self.G=nx.Graph(self.G)#nx.is_frozen(self.G) False
        # 去除自环
        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        nodes1 = list(self.G.nodes)
        # 重新编号
        self.G = self.rename_nodes(self.G)
        self.numNodes=self.G.number_of_nodes()
        self.numEdges=self.G.number_of_edges()
        nx.write_edgelist(self.G, self.edgelist_fname, data=False)
        print(f"{self.dataset_name}:  |V|={self.numNodes}  |E|={self.numEdges}",end="")
        # 处理节点类别文件
        if self.has_label:
            nodes2 = list(self.G.nodes)
            d_node = dict(zip(nodes1, nodes2))
            new_labels ={}
            d_class = {}
            labels = np.loadtxt(self.raw_label_fname, dtype=int, comments=None, delimiter=None).T
            labels = dict(zip(labels[0], labels[1])) 
            class_id = 0  # 对于label也重新索引
            for k, v in labels.items():
                if k not in d_node:
                    continue
                if v not in d_class:
                    d_class[v] = class_id
                    class_id += 1
                new_labels[d_node[k]] = d_class[v]
            res = np.array([list(new_labels.keys()), list(new_labels.values())]).T
            # self.node_label_dict = new_labels
            self.node_label_list = np.array(sorted(new_labels.items(),key=lambda x:x[0])).T[1]
            self.numLabels = max(self.node_label_list) + 1
            # np.savetxt(self.label_fname, res, "%d")
            np.save(self.label_fname,self.node_label_list)
            print(f" |labels|={self.numLabels}",end="")
        print()
        self.check_input()
        return

    def check_input(self):
        print("输入文件：", self.dataset_name)
        with open(self.edgelist_fname, 'r') as f:
            all = f.readlines()
        id2idx = {}
        adj = defaultdict(dict)
        edgelist = []
        num = 0
        maxid = -1
        for i, line in enumerate(all, 1):
            a, b = line.strip().split()
            a, b = int(a), int(b)
            if a > maxid:
                maxid = a
            if b > maxid:
                maxid = b
            if a not in id2idx:
                id2idx[a] = num
                num += 1
            if b not in id2idx:
                id2idx[b] = num
                num += 1
            _a, _b = a, b
            a, b = id2idx[a], id2idx[b]
            if a == b:
                print(f"line {i}:  {_a, _b}  存在自环")
                # exit()
            if b not in adj[a]:
                adj[a][b] = 1
            else:
                print(f"line {i}:  {_a, _b}  存在重复边")
                # exit()
            if a not in adj[b]:
                adj[b][a] = 1
            else:
                print(f"line {i}:  {_a, _b}  存在重复边")
                # exit()
        if maxid != num - 1:
            print('节点id不符合规范!')
            exit()
        print(f"输入文件是规范的：节点最大id（{maxid}）等于节点数（{num}）减1，无自环，无重复边")

    def largest_connected_component(self, H: nx.Graph):
        return H.subgraph(max(nx.connected_components(H), key=len))

    def rename_nodes(self, H: nx.Graph):
        # relabel nodes to integers indexed from 0 to n-1
        mapping = dict(zip(list(H.nodes()), range(len(H))))
        H = nx.relabel_nodes(H, mapping)
        return H


def run(path='./raw/email_eu.txt',l=1):
    args=Args()
    args.dataPath=path
    if l:
        args.labelPath=args.dataPath[:-4]+'_labels.txt'
    data=Data(args)
    data.preprocess()

# 当作无向图，取最大连通分量，去除自环和重复边


dataset_list=['email_eu',   ]
run(r'E:\Desktop\PythonProjects\SSumMpy\dataset\base\fb.txt',0)