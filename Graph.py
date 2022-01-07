from Node import Node


class Graph:

    q_identifier_list = []

    def __init__(self):

        self.nodes = []
        self.edges = []
        self.q_identifier_list = []

    def print_nodes(self):
        for node in self.nodes:
            node.print_info()

    def print_graph(self):
        self.print_nodes()
        print(self.edges)

    def add_node(self, node):
        self.nodes.append(node)

    def get_nodes(self):
        return self.nodes

    def take_node(self, id) -> Node:
        for n in self.nodes:
            if n.id == id:
                return n

    def check_roots(self):
        for node in self.nodes:
            root = True
            for edge in self.edges:
                if node.id == edge[1]:
                    root = False
            node.set_is_root(root)

    def get_parent(self, node):
        for edge in self.edges:
            if edge[1] == node.id:
                return self.take_node(edge[0])