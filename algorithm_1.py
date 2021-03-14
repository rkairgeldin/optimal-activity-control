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
                        'time_infect_recover': None,
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
                           'infect_infected': True,
                           }

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
                if (round(np.random.uniform(0,1), _DECIMAL) <= prob) and (j not in G[node]):

                    G.add_edge(node, j, perm=False, color='r', infect_carrier=True,
                               infect_infected=True)  # G is updated



        except:
            print("############EXCEPT############")  # exception when number of active nodes is lower than m
        active_nodes.append(node)  # active nodes are updated

    assert (type(active_nodes) == list)

    return G, active_nodes



def degree_temper(G, node): # degree of active nodes
    count = 0

    for i in G[node]:
        if not G[node][i]['perm']:
            count = count + 1
    return count


def active_neighbors(G, node): #list of active neighbors
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


def infect_neighbor(Graph, t, node, _DECIMAL):

    G = Graph.copy()
    if G.nodes[node]['state'] == 'C':
        for j in G[node]:
            if G[node][j]['infect_carrier']:

                # can no longer infect in this iteration
                G[node][j]['infect_carrier'] = False

                # infection time by node->j
                infect_t = round(t + generate_time(state_param[G.nodes[node]['state']]['beta'], _DECIMAL), _DECIMAL)
                if G.nodes[j]['time_infect']:
                    infect_t = min(G.nodes[j]['time_infect'], infect_t)


                #link duration between node and j

                if G[node][j]['perm']:
                    d_t = 100000
                else:
                    d_t = min(abs(t - G.nodes[j]['time_inactivate']),
                              abs(t - G.nodes[node]['time_inactivate']))
                    d_t = round(d_t, _DECIMAL)



                # node->recovery
                if G.nodes[node]['time_carrier_recover']:
                    if G.nodes[node]['time_carrier_recover'] > infect_t and d_t > infect_t:
                        # j is infected
                        G.nodes[j]['time_infect'] = infect_t
                # node->infection
                elif G.nodes[node]['time_carrier_infected']:
                    if G.nodes[node]['time_carrier_infected'] > infect_t and d_t > infect_t:
                        G.nodes[j]['time_infect'] = infect_t

    if G.nodes[node]['state'] == 'I':
        for j in G[node]:
            if G[node][j]['infect_infected']:

                # can no longer infect through this link
                G[node][j]['infect_infected'] = False

                # infection time by node->j
                infect_t = round(t + generate_time(state_param[G.nodes[node]['state']]['beta'], _DECIMAL), _DECIMAL)
                if G.nodes[j]['time_infect']:
                    infect_t = min(G.nodes[j]['time_infect'], infect_t)

                # link duration between node and j
                if G[node][j]['perm']:
                    d_t = 100000
                else:
                    d_t = min(abs(t - G.nodes[j]['time_inactivate']),
                              abs(t - G.nodes[node]['time_inactivate']))
                    d_t = round(d_t, _DECIMAL)

                # node->recovery
                if G.nodes[node]['time_infect_recover'] > infect_t and d_t > infect_t:
                    # j is infected
                    G.nodes[j]['time_infect'] = infect_t

    return G

def carrier_next_state(Graph, t, node, _DECIMAL):

    G = Graph.copy()
    if G.nodes[node]['state'] == 'C':
        #carrier node either recovers or gets infected
        #runs from carrier state to I or R
        if G.nodes[node]['time_carrier_recover'] == None and G.nodes[node]['time_carrier_infected'] == None:
            t_i = generate_time(state_param[G.nodes[node]['state']]['nu_I'], _DECIMAL) #time when the node becomes infected
            t_r = generate_time(state_param[G.nodes[node]['state']]['nu_R'], _DECIMAL) #time when the node recovers
            if t_i - t_r > 0:
                G.nodes[node]['time_carrier_recover'] = round(t + t_r, _DECIMAL)
            else:
                G.nodes[node]['time_carrier_infected'] = round(t + t_i, _DECIMAL)
    return G

def infected_next_state(Graph, t, node, _DECIMAL):

    G = Graph.copy()
    if G.nodes[node]['state'] == 'I' and G.nodes[node]['time_infect_recover']==None:
        G.nodes[node]['time_infect_recover'] = round(t + generate_time(state_param[G.nodes[node]['state']]['gamma'], _DECIMAL), _DECIMAL)
    return G

def print_graph(G, pos, new_folder, t):
    colors = [G[x][y]['color'] for x, y in G.edges]
    labels = {i: G.nodes[i]['state']+str(' ')+str(i) for i in G.nodes}
    plt.figure()
    plt.clf()
    nx.draw(G, pos=pos, edge_color=colors, labels=labels)
    plt.savefig(new_folder+"/Simulation_progress at "+str(t)+".png", dpi=300)
    plt.close()

def parsing():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=None, type=int, help='Seed for random number generator')
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
        G = nx.erdos_renyi_graph(args.nodes, args.probability, seed=args.seed, directed=False)
    elif args.command == 'BA':
        G = nx.barabasi_albert_graph(args.nodes, args.pref_edges, seed=args.seed, directed=False)
    elif args.command == 'WS':
        G = nx.watts_strogatz_graph(args.nodes, args.knearest, args.probability, seed=args.seed, directed=False)

    T = args.time
    show_progress = args.verbose

    return G, T, show_progress



if __name__ == '__main__':

    G, T, show_progress = parsing()


    pos = nx.circular_layout(G)

    path = os.getcwd()
    new_folder = path + '/progress_algorithm_1'
    if os.path.exists(new_folder):
        shutil.rmtree(new_folder)
        os.makedirs(new_folder)
    else:
        os.makedirs(new_folder)



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

    G = initialize_attributes(G, state_param, S, DECIMAL)

    result_dict = dict()
    active_nodes = list()  # list of active nodes


    step_res = round(1 / 10 ** DECIMAL, DECIMAL)

    for t in np.arange(0, T, step_res):
        t = round(t, DECIMAL)
        current_state_count = {'S': 0, 'C': 0, 'I': 0, 'R': 0}

        for i in G.nodes:

            if G.nodes[i]['state'] == 'S' and G.nodes[i]['time_infect'] == t:
                G.nodes[i]['state'] = 'C'
            if G.nodes[i]['state'] == 'C' and G.nodes[i]['time_carrier_recover'] == t:
                G.nodes[i]['state'] = 'R'
            if G.nodes[i]['state'] == 'C' and G.nodes[i]['time_carrier_infected'] == t:
                G.nodes[i]['state'] = 'I'
            if G.nodes[i]['state'] == 'I' and G.nodes[i]['time_infect_recover'] == t:
                G.nodes[i]['state'] = 'R'

            G, active_nodes = active_nodes_update(G, t, i, active_nodes, prob, DECIMAL)  # active nodes
            G, active_nodes = inactive_nodes_update(G, t, i, active_nodes, state_param, DECIMAL)  # inactive nodes

            G = carrier_next_state(G, t, i, DECIMAL)  # C->I C->R
            G = infected_next_state(G, t, i, DECIMAL)
            G = infect_neighbor(G, t, i, DECIMAL)  # infecting neighbor nodes

            current_state_count[G.nodes[i]['state']] = current_state_count[G.nodes[i]['state']] + 1

        result_dict[t] = current_state_count

        if show_progress:
            print_graph(G,pos, new_folder, t)


    plt.plot(list(result_dict.keys()), [i['S'] for i in result_dict.values()], color='red', label='Susceptible')
    plt.plot(list(result_dict.keys()), [i['C'] for i in result_dict.values()], color='green', label='Carrier')
    plt.plot(list(result_dict.keys()), [i['I'] for i in result_dict.values()], color='blue', label='Infected')
    plt.plot(list(result_dict.keys()), [i['R'] for i in result_dict.values()], color='black', label='Recovered')
    plt.legend()
    plt.savefig("algorithm_1_SIRC.png", dpi=300)
    plt.show()
    plt.close()
