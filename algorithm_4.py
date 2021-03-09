import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.manifold import TSNE
import shutil
import os


def save_folder(text):
    path = os.getcwd()
    new_folder = path + '\\'+text
    if os.path.exists(new_folder):
        shutil.rmtree(new_folder)
        os.makedirs(new_folder)
    else:
        os.makedirs(new_folder)

    return new_folder


DIMS = 20 #embedding size
W_L = 16 #walk length
N_W = 30 #number of walks
W = 2 #workers
WINDOW = 10 #window
MIN_COUNT = 1 #min count

embedding_folder = save_folder('graph_embeddings')

for i in range(1):
    G = nx.read_adjlist('graph_samples_algorithm_3/DTDG_'+str(i)+'_'+str(0)+'.adjlist', nodetype=int)
    #print('graph_samples_algorithm_3/DTDG_'+str(i)+'_'+str(0)+'.edgelist')
    with open('graph_samples_algorithm_3/DTDG_'+str(i)+'_'+str(0)+'node_list.txt','r') as f:
        state = f.readlines()
        #print(state)
        attrs = {int(line.split(' ')[0]): {'state': line.split(' ')[1][0]} for line in state}

    nx.set_node_attributes(G, attrs)
    #print('graph_samples_algorithm_3/DTDG_'+str(i)+'_'+str(0)+'node_list.txt')


    n2v = Node2Vec(G, dimensions=DIMS, walk_length=W_L, num_walks=N_W, workers=W)
    model = n2v.fit(window=WINDOW, min_count=MIN_COUNT)

    nodes = [x for x in model.wv.vocab]

    embeddings = np.array([model.wv[x] for x in nodes])

    tsne = TSNE(n_components=2, perplexity=10)  # 15
    embeddings_2d = tsne.fit_transform(embeddings)
    figure = plt.figure(figsize=(11, 9))

    ax = figure.add_subplot(111)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    for num, x in enumerate(nodes):
        ax.annotate(x + '_' + G.nodes[int(x)]['state'], (embeddings_2d[num, 0], embeddings_2d[num, 1]))
    plt.savefig(embedding_folder+'/DTDG_'+str(i)+'_'+str(0)+'.png', dpi=300)
    plt.show()