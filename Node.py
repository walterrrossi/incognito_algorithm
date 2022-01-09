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
        Class Node initialization function.

        Args:
            marked (bool): boolean value indicating whether the node is marked k-anonymous
            q_identifiers (list): list with QI ids to be generalized
            generalization_level (list): list of generalization levels
        """
        self.id = self._ID
        self.__class__._ID += 1
        self.marked = marked
        self.q_identifiers_list = q_identifiers
        self.generalization_level = generalization_level


    def set_is_root(self, is_root):
        """
        Function to set the Node as root.

        Args:
            is_root (bool): boolean value indicating whether the node is root
        """
        self.is_root = is_root


    def set_marked(self, marked):
        """
        Function to mark the Node.

        Args:
            marked (bool): boolean value indicating whether the node is marked
        """
        self.marked = marked


    def set_q_identifiers_list(self, q_identifiers_list):
        """
        Setter of the QI-list.

        Args:
            q_identifiers_list (list): list with QI ids to be anonymized
        """
        self.q_identifiers_list = q_identifiers_list


    def set_gen_level(self, generalization_level):
        """
        Setter of the generalization levels.

        Args:
            generalization_level (list): list of generalization levels
        """
        self.generalization_level = generalization_level


    def print_info(self):
        """
        Function to show informations about the Node.

        """
        print("id:" + str(self.id)+" " + str(self.q_identifiers_list) +
              " " + str(self.generalization_level) + " "+ str(self.marked))
    
