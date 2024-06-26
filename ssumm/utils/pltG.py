import textwrap
import random

import networkx as nx
from matplotlib import pyplot as plt

def pltG(G):
    # nx.draw(G)
    # plt.show()
    # 可以直接这样画 nx.draw(G, with_labels=True, node_color='white')

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='w')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)

    # l=[[(a[0],a[1]),a[2]['weight']] for a in list(G.edges(data=True))]
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=dict(l))
    # nx.draw_networkx_edge_labels(G, pos)

    # plt.savefig(f'g.jpg')
    # plt.clf()
    plt.show()


def gen_random_Graph(n=500, p=0.8):
    # p=n/(n*(n-1)/2)
    G = nx.gnp_random_graph(n, p)

    return G


# nx.write_edgelist(nx.path_graph(4), "test.edgelist")
# G=nx.read_edgelist('test.txt',create_using=nx.Graph,nodetype=int,data=(("weight", float),))
G=gen_random_Graph(10,0.3)
pltG(G)

def plt_Gs(dataset_name, figsize=12):
    dataset_folder = '../dataset/'
    output_folder = '../result/'

    path1 = f'output/{dataset_name}/suedges.txt'
    path2 = f'output/{dataset_name}/snlist.txt'
    G = nx.Graph()

    edges = []
    with open(path1, 'r') as f:
        a = f.readlines()
    for i in a:
        x, y = i.strip().split()
        edges.append((int(x), int(y)))
    G.add_edges_from(edges)

    snlist = {}
    labels = {}
    highlight_nodes = []
    with open(path2, 'r') as f:
        a = f.readlines()
    for i in a:
        i = [int(v) for v in i.strip().split('\t')]
        if len(i) > 2:
            snlist[i[0]] = f'{i[1:]}'
            highlight_nodes.append(i[0])
            labels[i[0]] = textwrap.fill(f'{",".join([str(_) for _ in i[1:]])}', 4)
        else:
            labels[i[0]] = f'{i[0]}'

    # 可以直接这样画
    # nx.draw(G, with_labels=True, node_color='white')

    # 也可以突出一些节点！
    plt.figure(figsize=(figsize, figsize))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='w')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels=labels)
    nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes, node_color='r', node_size=300)  # node_size=200,

    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)

    s = ''
    for i, v in snlist.items():
        s += str(i) + ':' + v + '\n'

    plt.legend([s], loc='upper right')
    # plt.subplots_adjust(wspace=0.3)
    plt.savefig(f'output/{dataset_name}/gs.jpg')
    plt.clf()


exit()