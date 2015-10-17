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

    # You can set up static state here
    stations = []

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

    # Checks if we can use a given path
    def path_is_valid(self, state, path):
        graph = state.get_graph()
        for i in range(0, len(path) - 1):
            if graph.edge[path[i]][path[i + 1]]['in_use']:
                return False
        return True

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

        graph = state.get_graph()
        station = graph.nodes()[0]

        self.update_weights(state)

        commands = []
        if not self.stations:
            commands.append(self.build_command(station))
            self.stations.append(station)

        pending_orders = state.get_pending_orders()
        if len(pending_orders) != 0:
            order = max(pending_orders, key = lambda o: o.get_money())
            path = nx.shortest_path(graph, station, order.get_node())
            if self.path_is_valid(state, path):
                commands.append(self.send_command(order, path))

        return commands
