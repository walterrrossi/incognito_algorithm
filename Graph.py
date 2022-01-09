from Node import Node


class Graph:

    q_identifier_list = []


    def __init__(self):
        """
        Class Graph initialization function.

        """
        self.nodes = []
        self.edges = []
        self.q_identifier_list = []


    def print_nodes(self):
        """
        Function to show informations about the nodes inside the Graph.

        """
        for node in self.nodes:
            node.print_info()


    def print_graph(self):
        """
        Function to show informations about the Graph.

        """
        self.print_nodes()
        print(self.edges)


    def add_node(self, node):
        """
        Function to add a node to the Graph.

        Args:
            node (Node): object node to be added to the Graph
        """
        self.nodes.append(node)


    def get_nodes(self):
        """
        Function to get the list of nodes of the Graph.

        """
        return self.nodes


    def take_node(self, id) -> Node:
        """
        Function to get the node object given its id.

        Args:
            id (integer): numeric id of the node to find
        
        Returns:
            [Node]: the node object with the specified id
        """
        for n in self.nodes:
            if n.id == id:
                return n


    def check_roots(self):
        """
        Function to set roots nodes inside the Graph.

        """
        for node in self.nodes:
            root = True
            for edge in self.edges:
                if node.id == edge[1]:
                    root = False
            node.set_is_root(root)


    def get_parent(self, node) -> Node:
        """
        Function to get the parent of a node.

        Args:
            node (Node): the node (child) whose parent you want to find 
        
        Returns:
            [Node]: the node-object of the parent
        """
        for edge in self.edges:
            if (edge[1] == node.id):
                return self.take_node(edge[0])

