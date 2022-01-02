from Node import Node

class Graph:

    
    q_identifier_list = []

    def __init__(self):
        
        self.nodes = []
        self.edges = []
        self.q_identifier_list= []

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