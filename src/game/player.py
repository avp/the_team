import logging as log
import time
import math
import logging as log
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
        x = float(x)
        sigma = float(sigma)
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
        self.total_weight = 0

        self.gaussians = []
        self.stations = []

        for i in xrange(int(ORDER_VAR * 3)):
            self.gaussians.append(self.gaussian(ORDER_VAR ** 0.5, i))

        for n in graph.nodes():
            self.g.node[n]['weight'] = 1.0 / graph.number_of_nodes()

        self.more_stations = True
        self.dists = nx.shortest_path_length(self.g)

    def update_weights(self, state):
        graph = state.get_graph()
        for order in state.pending_orders:
            if order.get_time_created() == state.get_time():
                frontier = set([order.get_node()])
                new_front = set()
                seen = set(frontier)
                for i in xrange(int(ORDER_VAR * 2)):
                    for n in frontier:
                        self.g.node[n]['weight'] += self.gaussians[i]
                        for nbr in self.g.neighbors(n):
                            seen.add(nbr)
                            new_front.add(nbr)
                frontier = new_front

        self.total_weight = sum(self.g.node[n]['weight'] for n in graph.nodes())
        # for n in graph.nodes():
        #     self.g.node[n]['weight'] /= total

    def fitness(self, sample, weight=None):
        if not self.stations:
            return float("-inf")
        s = 0
        for node in sample:
            d = min( (self.dists[st][node] for st in self.stations) )
            s += (SCORE_MEAN - (d * DECAY_FACTOR)) * (self.g.node[node]['weight'] / self.total_weight)
        return s * (GAME_LENGTH - self.state.get_time() - 1) * ORDER_CHANCE

    # Checks if we can use a given path
    def path_is_valid(self, state, path):
        graph = state.get_graph()
        for i in range(0, len(path) - 1):
            if graph.edge[path[i]][path[i + 1]]['in_use']:
                return False
        return True

    def set_path(self, g, path, attr, val):
        for i in range(0, len(path) - 1):
            g.edge[path[i]][path[i + 1]][attr] = val

    def get_max_weight(self, graph, attr='weight'):
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
        #log.warning("L1")
        self.state = state
        money = state.money
        graph = state.get_graph()

        t0 = time.time()
        self.update_weights(state)
        #log.warning("L1.5")

        for (u, v) in self.g.edges():
            self.g.edge[u][v]['free'] = float('inf') if self.state.graph.edge[u][v]['in_use'] else 1

        #log.warning("L2")
        commands = []
        if (not self.stations) and state.pending_orders:
            newstation = self.get_max_weight(graph)
            commands.append(self.build_command(newstation))
            self.stations.append(newstation)
            money -= INIT_BUILD_COST

        t1 = time.time()
        stationcost = INIT_BUILD_COST * (BUILD_FACTOR ** len(self.stations))
        if stationcost <= money and self.more_stations:
            size = 250
            if graph.number_of_nodes() > size:
                sample = random.sample(graph.nodes(), size)
            else:
                sample = graph.nodes()
            oldfitness = self.fitness(sample)
            maxdelta = 0
            best_station = None
            for newstation in sample:
                if newstation in self.stations:
                    continue
                self.stations.append(newstation)
                newfitness = self.fitness(sample)
                self.stations.pop()
                delta = newfitness - oldfitness
                if delta > maxdelta and delta > stationcost:
                    best_station = newstation
                    maxdelta = delta
            if best_station:
                commands.append(self.build_command(best_station))
                self.stations.append(best_station)

            else:
                self.more_stations = False

        #log.warning("L3")
        pending_orders = set(state.get_pending_orders())
        t2 = time.time()

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
                    path = nx.shortest_path(self.g, station, target, weight='free')
                    if not self.path_is_valid(state, path): continue
                    score = o_val-len(path)*DECAY_FACTOR
                    if score > best_score:
                        best_score = score
                        best_path = path
                        best_order = order

            if best_score > 0:
                paths.append((best_path, best_order))
                self.set_path(self.g, best_path, 'free', float('inf'))
                self.set_path(state.graph, best_path, 'in_use', True)
                pending_orders.remove(best_order)
            else:
                break

        #log.warning("L4")
        for (path, order) in paths:
            # if self.path_is_valid(state, path):
            commands.append(self.send_command(order, path))
            # else:
            #     log.warning("WHAT THE HELLLLLLLLL" * 100)

        t3 = time.time()
        #log.warning("L5")
        log.warning("%.5f, %.5f, %.5f", t1 - t0, t2 - t1, t3 - t2)


        return commands
