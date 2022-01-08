from pandas.core.frame import DataFrame


class Node:
    
    _ID = 0
    is_root = False
    marked = False
    q_identifiers_list = []
    generalization_level = []
    frequency_set = DataFrame
    parent1 = 0
    parent2 = 0

    def __init__(self, marked, q_identifiers, generalization_level):
        """
        Funzione di inizializzazione classe nodo

        Args:
            marked (bool): valore booleano che indica se il nodo è k-anonimo
            q_identifiers (list): lista con id dei QI da anonimizzare
            generalization_level (list): lista dei livelli di generalizzazione
        """
        self.id = self._ID
        self.__class__._ID += 1
        self.marked = marked
        self.q_identifiers_list = q_identifiers
        self.generalization_level = generalization_level

    def set_is_root(self, is_root):
        self.is_root = is_root

    def set_marked(self, marked):
        self.marked = marked

    def set_q_identifiers_list(self, q_identifiers_list):
        self.q_identifiers_list = q_identifiers_list

    def set_gen_level(self, generalization_level):
        self.generalization_level = generalization_level

    def print_info(self):
        print("id:" + str(self.id)+" " + str(self.q_identifiers_list) +
              " " + str(self.generalization_level) + " "+ str(self.marked))
    
