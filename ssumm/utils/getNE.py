import os
import torch
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

class Args:
    pass



def getNodeEmb(args):
    args.datapath = './dataset/' + args.dataset_name + '.edgelist'
    args.labelpath='./dataset/' + args.dataset_name + '.label.npy'
    args.embedding_folder = "./output_emb/"
    os.makedirs(args.embedding_folder, exist_ok=True)
    args.embedding_path = args.embedding_folder + args.dataset_name + '_' + args.model + '.npy'

    G=nx.read_edgelist(args.datapath,nodetype=int)
    A=nx.to_numpy_array(G, nodelist=sorted(G.nodes))#
    edgelist = list(G.edges()) + [(b, a) for (a, b) in list(
        G.edges())]
    edge_index = torch.tensor(sorted(edgelist), dtype=torch.long).t().contiguous()

    args.G=G
    args.A=A
    args.Acsr=csr_matrix(A)
    args.edge_index=edge_index
    args.numNodes = A.shape[0]


    if 0:#os.path.exists(args.embedding_path):
        print('[ load OK! ]')
        return np.load(args.embedding_path)
    if args.model=='gae':#flag model
        pass
    elif args.model=='gcncd':
        args.seed = 2023
        args.node_label_list = np.load(args.labelpath)
        from CPNE.gcn import create_gcncd
        res=create_gcncd(args)
    elif args.model=='mnmf':
        from karateclub import MNMF
        model = MNMF(clusters=args.clusters, dimensions=args.dimensions, iterations=args.iterations)
        model.fit(args.G)
        res=model.get_embedding()
    else:
        print("error")
        exit()
    np.save(args.embedding_path, res)
    print(f"embedding of {args.dataset_name}:  {res.shape}")
    return res






args=Args()
args.dataset_name='ka'
args.model='gcncd'

if args.model=='mnmf':
    args.clusters=42
    args.dimensions=50
    args.iterations=200
    getNodeEmb(args)

elif args.model=='gcncd':
    args.label_type = 'class'
    args.features_type = 'eye'
    args.eigdim=3
    args.dimensions = 10
    getNodeEmb(args)