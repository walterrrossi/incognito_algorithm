class Node:

    is_root = False
    marked = False
    q_identifier_list = []
    generalization_level = []

    def __init__(self, marked, q_identifiers, generalization_level):
        self.marked = marked
        self.q_identifier_list = q_identifiers
        self.generalization_level = generalization_level
     
    def set_is_root(self, is_root):
        self.is_root = is_root
    
    def set_marked(self, marked):
        self.marked = marked
    
    def set_q_identifier_list(self, q_identifier_list):
        self.q_identifier_list = q_identifier_list
    
    def set_gen_level(self, generalization_level):
        self.generalization_level = generalization_level