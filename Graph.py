from Node import Node

class Graph:

    nodes = dict()
    edges = dict()
    q_identifier_list = []

    def __init__(self):
        self.nodes = dict()
        self.edges = dict()
        self.q_identifier_list= []

    def initialize(self, q_identifiers, generalization_level):
        for qi in q_identifiers:
            self.q_identifier_list.append(qi)
            for i in range(0, generalization_level[qi]):
                n = Node(False, qi, i)
                if i == 0:
                    n.set_is_root(True)
                self.nodes[i] = n
            if generalization_level[qi] >= 1:
                for i in range(1, generalization_level[qi]):
                    self.edges[i-1] = i
        print(self.nodes)
        print(self.edges)