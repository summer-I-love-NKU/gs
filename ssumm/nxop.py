# import networkx as nx
# G=nx.karate_club_graph()
# labels = {node: G.nodes[node]['club'] for node in G.nodes}
# nx.write_edgelist(G,'ka.txt',data=False)
# with open('ka_labels.txt','w') as f:
#     num={'Mr. Hi':0,'Officer':1}
#     for k,v in labels.items():
#         f.write(f'{k} {num[v]}\n')
#
# a=1

import networkx as nx
G=nx.karate_club_graph()
eigenvector_centrality = nx.eigenvector_centrality(G)
sorted_eigenvector_centrality = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)
print(sorted_eigenvector_centrality)
