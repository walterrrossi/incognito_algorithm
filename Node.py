class Node:

    id = 0
    is_root = False
    marked = False
    q_identifier_list = list()
    generalization_level = []
    parent1 = 0
    parent2 = 0

    def __init__(self, id, marked, q_identifiers, generalization_level):
        #self.uid+=1
        self.id=id
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

    def print_info(self):
        print("id:"+ str(self.id)+" "+ str(self.q_identifier_list) + " "+ str(self.generalization_level))
    