import logging as log
import math
import networkx as nx
import random
from base_player import BasePlayer
from settings import *

class Player(BasePlayer):
    """
    You will implement this class for the competition. DO NOT change the class
    name or the base class.
    """

    def gaussian(self, sigma, x):
        """ Gaussian, mean=0, sigma=sigma """
        sqrt2pi = math.sqrt(2*math.pi)
        expterm = math.exp(-(x**2) / (2 * sigma**2))
        return 1.0 / (sigma * sqrt2pi) * expterm

    def __init__(self, state):
        """
        Initializes your Player. You can set up persistent state, do analysis
        on the input graph, engage in whatever pre-computation you need. This
        function must take less than Settings.INIT_TIMEOUT seconds.
        --- Parameters ---
        state : State
            The initial state of the game. See state.py for more information.
        """
        graph = state.get_graph()

        self.g = graph.copy()

        self.gaussians = []
        self.stations = []

        for i in xrange(int(ORDER_VAR * 3)):
            self.gaussians.append(self.gaussian(ORDER_VAR, i))

        for n in graph.nodes():
            self.g.node[n]['weight'] = 1.0 / graph.number_of_nodes()

    def update_weights(self, state):
        graph = state.get_graph()
        for order in state.pending_orders:
            if order.get_time_created() == state.get_time():
                frontier = [order.get_node()]
                for i in xrange(int(ORDER_VAR * 2)):
                    for n in frontier:
                        self.g.node[n]['weight'] += self.gaussians[i]
                    frontier = set(sum([self.g.neighbors(n) for n in frontier], []))

        total = 0.0
        for n in graph.nodes():
            total += self.g.node[n]['weight']
        for n in graph.nodes():
            self.g.node[n]['weight'] /= total
        return

    def fitness(self, weight=None):
        dists = {(station,dest): val for station in self.stations for (dest,val) in
                    nx.shortest_path_length(self.g, station, weight=weight).iteritems() }
        s = 0
        for node in self.g.nodes():
            d = min( (dists[(st, node)] for st in self.stations) )
            s += (SCORE_MEAN - (d * DECAY_FACTOR)) * self.g.node[node]['weight']
        return s * (GAME_LENGTH - self.state.get_time() - 1) * ORDER_CHANCE

    # Checks if we can use a given path
    def path_is_valid(self, state, path):
        graph = state.get_graph()
        for i in range(0, len(path) - 1):
            if graph.edge[path[i]][path[i + 1]]['in_use']:
                return False
        return True

    def set_path(self, path, val):
        for i in range(0, len(path) - 1):
            self.g.edge[path[i]][path[i + 1]]['free'] = val

    def get_max_node(self, graph, attr):
        result = graph.nodes()[0]
        for n in graph.nodes():
            if self.g.node[n][attr] > self.g.node[result][attr]:
                result = n
        return result

    def step(self, state):
        """
        Determine actions based on the current state of the city. Called every
        time step. This function must take less than Settings.STEP_TIMEOUT
        seconds.
        --- Parameters ---
        state : State
            The state of the game. See state.py for more information.
        --- Returns ---
        commands : dict list
            Each command should be generated via self.send_command or
            self.build_command. The commands are evaluated in order.
        """

        # We have implemented a naive bot for you that builds a single station
        # and tries to find the shortest path from it to first pending order.
        # We recommend making it a bit smarter ;-)
        self.state = state
        graph = state.get_graph()

        self.update_weights(state)
        for (u, v) in self.g.edges():
            self.g.edge[u][v]['free'] = float('inf') if self.state.graph.edge[u][v]['in_use'] else 1

        commands = []
        if not self.stations:
            newstation = self.get_max_node(graph, 'weight')
            commands.append(self.build_command(newstation))
            self.stations.append(newstation)

        stationcost = INIT_BUILD_COST * BUILD_FACTOR ** len(self.stations)
        if stationcost <= state.money:
            newstation = self.get_max_node(graph, 'weight')
            oldfitness = self.fitness()
            self.stations.append(newstation)
            newfitness = self.fitness()
            if newfitness > oldfitness and stationcost < state.money:
              commands.append(self.build_command(newstation))
            else:
              self.stations.pop()

        pending_orders = set(state.get_pending_orders())

        paths = []
        ## Calculate paths
        while True:
            best_path = None
            best_order = None
            best_score = float("-inf")
            for order in pending_orders:
                o_val = state.money_from(order)
                target = order.get_node()
                for station in self.stations:
                    if nx.shortest_path_length(self.g, station, target, weight='free') > 3000:
                        continue
                    for path in nx.all_shortest_paths(self.g, station, target, weight='free'):
                        score = o_val-len(path)*DECAY_FACTOR
                        if score > best_score:
                            best_score = score
                            best_path = path
                            best_order = order

            if best_score > 0:
                paths.append((best_path, best_order))
                self.set_path(best_path, float('inf'))
                pending_orders.remove(best_order)
            else:
                break

        for (path, order) in paths:
            if self.path_is_valid(state, path):
                commands.append(self.send_command(order, path))
            else:
                log.warning("WHAT THE HELLLLLLLLL" * 100)


        return commands
