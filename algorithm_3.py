import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import argparse
import os
import shutil
from parameter_list import *


def generate_time(dist, _DECIMAL):
    delta_t = np.random.exponential(dist)
    if delta_t < 5/(10**(_DECIMAL+1)):
        delta_t = 1/(10**_DECIMAL)

    return round(delta_t, _DECIMAL)


def initialize_attributes(Graph, state_param, infected_nodes, _DECIMAL):
    G = Graph.copy()

    # infected_nodes - list of infected nodes or number of infected nodes. In case of number it is randomly selected from thelist of nodes

    attr_dict = dict()
    edge_attr = dict()

    if type(infected_nodes) == int:
        infected_nodes_list = random.sample(G.nodes, int(infected_nodes))
    if type(infected_nodes) == list:
        infected_nodes_list = infected_nodes

    for i in G.nodes:
        attr_dict[i] = {'active': False,
                        'state': 'S',
                        'time_infect': None,
                        'time_carrier_recover': None,
                        'time_carrier_infected': None,
                        'time_infect_recover': None
                        }
        if i in infected_nodes_list:
            attr_dict[i]['state'] = 'C'

        attr_dict[i]['time_activate'] = generate_time(state_param[attr_dict[i]['state']]['sigma_1'], _DECIMAL)

        attr_dict[i]['time_inactivate'] = round(
            attr_dict[i]['time_activate'] + generate_time(state_param[attr_dict[i]['state']]['sigma_2'], _DECIMAL),
            _DECIMAL)

        assert (attr_dict[i]['time_activate'] < attr_dict[i]['time_inactivate'])

    nx.set_node_attributes(G, attr_dict)

    for edge in G.edges:
        edge_attr[edge] = {'perm': True,
                           'color': 'b',
                           'infect_carrier': True,
                           'infect_infected': True, }

    nx.set_edge_attributes(G, edge_attr)

    return G


def active_nodes_update(Graph, t, node, active_nodes, prob, _DECIMAL):
    G = Graph.copy()
    # G is a graph
    # t is a moment in time
    # node which node is analyzed
    # active_nodes list of active nodes in the moment

    if G.nodes[node]['time_activate'] == t:

        G.nodes[node]['active'] = True  # node is activated
        try:
            for j in active_nodes:
                if (round(np.random.uniform(0, 1), _DECIMAL) <= prob) and (j not in G[node]):
                    d_t = min(abs(G.nodes[node]['time_activate'] - G.nodes[j]['time_inactivate']),
                              abs(G.nodes[node]['time_activate'] - G.nodes[node]['time_inactivate']))
                    d_t = round(d_t, _DECIMAL)
                    G.add_edge(node, j, perm=False, color='r', T=d_t)  # G is updated




        except:
            print("############EXCEPT############")  # exception when number of active nodes is lower than m
        active_nodes.append(node)  # active nodes are updated

    assert (type(active_nodes) == list)

    return G, active_nodes


def degree_temper(G, node):  # degree of active nodes
    count = 0

    for i in G[node]:
        if not G[node][i]['perm']:
            count = count + 1
    return count


def active_neighbors(G, node):  # list of active neighbors
    active_neighbor_list = []
    for i in G.neighbors(node):
        if not G[node][i]['perm']:
            active_neighbor_list.append(i)

    return active_neighbor_list


def inactive_nodes_update(Graph, t, node, active_nodes, state_param, _DECIMAL):
    G = Graph.copy()

    rem_list = list()

    if G.nodes[node]['time_inactivate'] == t:

        G.nodes[node]['active'] = False  # node is inactive

        t_act = generate_time(state_param[G.nodes[node]['state']]['sigma_1'], _DECIMAL)
        t_deact = generate_time(state_param[G.nodes[node]['state']]['sigma_2'], _DECIMAL)

        G.nodes[node]['time_activate'] = round(t + t_act, _DECIMAL)
        G.nodes[node]['time_inactivate'] = round(G.nodes[node]['time_activate'] + t_deact, _DECIMAL)
        active_nodes.remove(node)

        for j in G[node]:
            if not G[node][j]['perm']:
                rem_list.append(j)
        G.remove_edges_from([(node, k) for k in rem_list])

    return G, active_nodes


def DTDG(Graph, sampling_rate, state_param, T, _DECIMAL, prob):
    # G - dynamic graph
    # sampling_rate - snapshot of the graph each sampling_rate
    # T - maximum time of simulation
    active_nodes = list()  # list of active nodes
    G = Graph.copy()
    G_dict = {}
    count = 0

    assert sampling_rate < T * 10 ** _DECIMAL, "Sampling rate higher than maximum time steps in simulation"

    step_res = round(1 / 10 ** _DECIMAL, _DECIMAL)

    for t in np.arange(0, T, step_res):
        t = round(t, _DECIMAL)

        for i in G.nodes:
            G, active_nodes = active_nodes_update(G, t, i, active_nodes, prob, _DECIMAL)  # active nodes
            G, active_nodes = inactive_nodes_update(G, t, i, active_nodes, state_param, _DECIMAL)  # inactive nodes

        if t == 0:
            G_dict[0] = G.copy()
        count = count + 1
        if count == sampling_rate:
            G_dict[t] = G.copy()
            count = 0

        for q, w in G.edges:
            if not G[q][w]['perm']:
                if G[q][w]['T'] - step_res > 0:
                    G[q][w]['T'] = round(G[q][w]['T'] - step_res, _DECIMAL)

    return G_dict


def print_graph(G, pos, new_folder, t, ind):
    colors = [G[x][y]['color'] for x, y in G.edges]
    labels = {i: G.nodes[i]['state']+str(' ')+str(i) for i in G.nodes}
    plt.figure()
    plt.clf()
    nx.draw(G, pos=pos, edge_color=colors, labels=labels)
    plt.savefig(new_folder+"/Simulation_progress at iter="+str(t)+"ind="+str(ind)+".png", dpi=300)
    plt.close()


def carrier_transfer(beta, delta_t, mul=1):
    prob = (1 - np.exp(-(1 / beta) * delta_t)) * mul
    return np.random.uniform(0, 1) <= prob


def carrier_infecting(Graph, node, state_param, sampling_rate, _DECIMAL):
    delta_t_l = round(sampling_rate / 10 ** _DECIMAL, _DECIMAL)
    G = Graph.copy()
    if G.nodes[node]['state'] == 'C':
        for i in G[node]:
            if G[node][i]['perm']:
                if G.nodes[i]['state'] == 'S' and carrier_transfer(state_param[G.nodes[node]['state']]['beta'],
                                                                   delta_t_l):
                    G.nodes[i]['state'] = 'C'
            else:
                if carrier_transfer(state_param[G.nodes[node]['state']]['beta'], G[node][i]['T']) and G.nodes[i][
                    'state'] == 'S':
                    G.nodes[i]['state'] = 'C'
    return G


def infected_infecting(Graph, node, state_param, sampling_rate, _DECIMAL):
    delta_t_l = round(sampling_rate / 10 ** _DECIMAL, _DECIMAL)
    G = Graph.copy()
    if G.nodes[node]['state'] == 'I':
        for i in G[node]:
            if G[node][i]['perm']:
                if G.nodes[i]['state'] == 'S' and carrier_transfer(state_param[G.nodes[node]['state']]['beta'],
                                                                   delta_t_l):
                    G.nodes[i]['state'] = 'C'
            else:
                if carrier_transfer(state_param[G.nodes[node]['state']]['beta'], G[node][i]['T'],
                                    (1 / state_param[G.nodes[node]['state']]['sigma_1']) / (
                                            (1 / state_param[G.nodes[node]['state']]['sigma_1']) + (
                                            1 / state_param[G.nodes[node]['state']]['sigma_2']))) and G.nodes[i][
                    'state'] == 'S':
                    G.nodes[i]['state'] = 'C'
    return G


def change_node_state(Graph, new_states):
    G = Graph.copy()
    for i in G.nodes:
        G.nodes[i]['state'] = new_states[i]

    return G


def next_state(Graph, sampling_rate, state_param, _DECIMAL):
    G = Graph.copy()
    delta_t_l = round(sampling_rate / 10 ** _DECIMAL, _DECIMAL)
    new_states = dict()

    for node in G.nodes:
        if G.nodes[node]['state'] == 'C':
            nu_i = state_param[G.nodes[node]['state']]['nu_I']
            nu_r = state_param[G.nodes[node]['state']]['nu_R']
            prob_C = 1 - np.exp(-(1 / nu_i + 1 / nu_r) * delta_t_l)
            prob_I = ((1 / nu_i) / (1 / nu_i + 1 / nu_r)) * np.exp(-(1 / nu_i + 1 / nu_r) * delta_t_l)
            # prob_R = nu_r / (nu_i + nu_r) * np.exp(-(1 / nu_i + 1 / nu_r) * delta_t_l)

            prob = np.random.uniform(0, 1)

            if prob < prob_C:
                new_states[node] = 'C'
            elif prob_C <= prob < prob_C + prob_I:
                new_states[node] = 'I'
            elif prob_C + prob_I <= prob:
                new_states[node] = 'R'


        elif G.nodes[node]['state'] == 'I':

            gamma_i = state_param[G.nodes[node]['state']]['gamma']
            prob = 1 - np.exp(-(1 / gamma_i) * delta_t_l)

            if np.random.uniform(0, 1) <= prob:
                new_states[node] = 'I'
            else:
                new_states[node] = 'R'

        else:
            new_states[node] = G.nodes[node]['state']

    return new_states


def parsing():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=None, type=int, help='Seed for random number generator')
    parser.add_argument('-sr', "--sampling_rate", type=int, required=True,
                        help='Collect snapshot of the graph each -sr timestamp')
    parser.add_argument('-run', '--total_run', type=int, required=True, help='The number of simulation runs')
    parser.add_argument('-T', "--time", type=int, required=True, help='The duration of the simulation')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='if true saves the progress of infection at each timestamp (default = False)')

    subparser = parser.add_subparsers(dest='command')
    ER = subparser.add_parser('ER')
    WS = subparser.add_parser('WS')
    BA = subparser.add_parser('BA')
    ER.add_argument('-n', '--nodes', type=int, required=True, help='Number of nodes in the graph')
    ER.add_argument('-p', '--probability', type=float, required=True, help='Probability of edge creation')

    BA.add_argument('-n', '--nodes', type=int, required=True, help='Number of nodes in the graph')
    BA.add_argument('-m', '--pref_edges', type=int, required=True,
                    help='Number of edges to attach from a new node to existing nodes')

    WS.add_argument('-n', '--nodes', type=int, required=True, help='Number of nodes in the graph')
    WS.add_argument('-k', '--knearest', type=int, required=True,
                    help='Each node is connected to k nearest neighbors in ring topology')
    WS.add_argument('-p', '--probability', type=float, required=True, help='The probability of rewiring each edge')

    args = parser.parse_args()

    if args.command == 'ER':
        G0 = nx.erdos_renyi_graph(args.nodes, args.probability, seed=args.seed, directed=False)
    elif args.command == 'BA':
        G0 = nx.barabasi_albert_graph(args.nodes, args.pref_edges, seed=args.seed, directed=False)
    elif args.command == 'WS':
        G0 = nx.watts_strogatz_graph(args.nodes, args.knearest, args.probability, seed=args.seed, directed=False)

    T = args.time
    show_progress = args.verbose
    total = args.total_run  # Q
    sampling_rate = args.sampling_rate  # sampling_time


    return G0, T, show_progress, total, sampling_rate

def save_folder(text):
    path = os.getcwd()
    new_folder = path + '/'+text
    if os.path.exists(new_folder):
        shutil.rmtree(new_folder)
        os.makedirs(new_folder)
    else:
        os.makedirs(new_folder)

    return new_folder

def record_graph(Gr, graph_folder, t):
    nx.write_adjlist(Gr, graph_folder + "/DTDG_"+str(t)+"_"+str(0)+".adjlist")
    #nx.write_edgelist(Gr, graph_folder + "/DTDG_"+str(t)+"_"+str(ind)+".edgelist")
    f = open(graph_folder+"/DTDG_"+str(t)+"_"+str(ind)+"node_list.txt", "w")
    for nd in Gr.nodes:
        f.write(str(nd)+' '+Gr.nodes[nd]['state'])
        f.write('\n')
    f.close()

def graphsPreprocessing(N, folder, G, l, t):
    #G = nx.erdos_renyi_graph(10, 0.1, seed=None, directed=False)
    n = len(G.nodes)
    mapping = {'S':'0', 'C': '1', 'I': '2', 'R': '3'}
    with open(folder+"/_DTDG_Graph.txt", "a") as f:
        if t==0:
            f.write(str(N)+'\n')
        f.write(str(n)+" "+str(l)+'\n')
        for nd in G.nodes:
            n_list = [str(k) for k in G.neighbors(nd)]
            m = len(n_list)
            #f.write(str(nd)+' '+G.nodes[nd]['state'])
            f.write(mapping[str(G.nodes[nd]['state'])]+" "+str(m)+" "+" ".join(n_list))
            f.write('\n')

if __name__ == '__main__':

    G0, T, show_progress, total, sampling_rate = parsing()
    pos = nx.circular_layout(G0)

    new_folder = save_folder('progress_algorithm_3')
    graph_folder = save_folder('graph_samples_algorithm_3')
    processed_file_folder = save_folder('processed_data')

    state_param = {'S': {'sigma_1': 1 / parameters['sigma_1_H'],
                         'sigma_2': 1 / parameters['sigma_2_H']
                         },

                   'C': {'sigma_1': 1 / parameters['sigma_1_H'],
                         'sigma_2': 1 / parameters['sigma_2_H'],
                         'beta': 1 / parameters['beta_C'],
                         'nu_I': 1 / parameters['nu_I'],
                         'nu_R': 1 / parameters['nu_R']},
                   'I': {'sigma_1': 1 / parameters['sigma_1_I'],
                         'sigma_2': 1 / parameters['sigma_2_I'],
                         'beta': 1 / parameters['beta_I'],
                         'gamma': 1 / parameters['gamma']},
                   'R': {'sigma_1': 1 / parameters['sigma_1_H'],
                         'sigma_2': 1 / parameters['sigma_2_H']}

                   }

    assert max(S) < max(G0.nodes), "S contain nodes that are not part of graph G0"
    G = initialize_attributes(G0, state_param, S, DECIMAL)

    cols = int(T*10**DECIMAL/sampling_rate+1)

    node_info = {'S': np.zeros((total, cols)),
                 'C': np.zeros((total, cols)),
                 'I': np.zeros((total, cols)),
                 'R': np.zeros((total, cols))}

    for ind in range(total):

        G_dict = DTDG(G, sampling_rate, state_param, T, DECIMAL, prob)
        new_states = {}

        for t, Gr in enumerate(G_dict.values()):

            try:
                Gr = change_node_state(Gr, new_states)
            except:
                print("First iteration")
            if ind == 0:
                
                graphsPreprocessing(len(G_dict), processed_file_folder, Gr, 0, t)
                record_graph(Gr, graph_folder, t)

            for i in S:
                Gr = carrier_infecting(Gr, i, state_param, sampling_rate, DECIMAL)
                Gr = infected_infecting(Gr, i, state_param, sampling_rate, DECIMAL)

            new_states = next_state(Gr, sampling_rate, state_param, DECIMAL)
            S.clear()

            for node, state in new_states.items():
                if state == 'C' or state == 'I':
                    S.append(node)

            for i in Gr.nodes:
                node_info[Gr.nodes[i]['state']][ind, t] = node_info[Gr.nodes[i]['state']][ind, t] + 1


            if show_progress:
                print_graph(Gr, pos, new_folder, t, ind)



            # print_graph(Graph, pos)
    inf_car_t = (node_info['C']+node_info['I'])/len(G0.nodes)
    inf_car_aver = inf_car_t.sum(axis=0)/total

    analysis = {'S': node_info['S'].sum(axis=0) / total,
                'C': node_info['C'].sum(axis=0) / total,
                'I': node_info['I'].sum(axis=0) / total,
                'R': node_info['R'].sum(axis=0) / total}

    plt.figure()

    plt.plot([z for z in G_dict.keys()], analysis['S'], color='red', label='Susceptible')
    plt.plot([z for z in G_dict.keys()], analysis['C'], color='green', label='Carrier')
    plt.plot([z for z in G_dict.keys()], analysis['I'], color='blue', label='Infected')
    plt.plot([z for z in G_dict.keys()], analysis['R'], color='black', label='Recovered')
    plt.legend()
    plt.savefig("algorithm_3_SIRC.png", dpi=300)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot([z for z in G_dict.keys()], inf_car_aver)
    plt.savefig("algorithm_3_SIRC_disease_prevalence", dpi=300)
    plt.close()
