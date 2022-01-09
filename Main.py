"""
    INCOGNITO_ALGORITHM
    
    K-anonymize a Dataset using Incognito Algorithm.

    - Main.py

    Summary:
        Core of the algorithm and main functions.

    Authors:
        Alessio Formica, Walter Rossi, Riccardo Poli.
        Unige, Data Protection & Privacy.

    Theoretical Sources:
        https://www.researchgate.net/publication/221213050_Incognito_Efficient_Full-Domain_K-Anonymity
        By Le Fevre, DeWitt, Ramakrishnan   
"""

import pandas as pd
import copy
import time
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame
from generalization import create_generalization_hierarchies
from generalization import generalize_data
from Graph import Graph
from Node import Node

# -------------------------------------------------------------------


def _log(content, enabled=True, endl=True):
    """
    Prints a log message.

    Args:
        content: Content of the message
        enabled (bool): If False the message is not printed
        endl (bool): If false write on the same line
    """

    if enabled:
        if endl:
            print(content)
        else:
            sys.stdout.write('\r' + content)


def initialize_graph(q_identifiers_id_lev_dict):
    """
    This function initializes the first graph of the algorithm. 
    This graph must contain as nodes the individual quasi identifiers with their levels of generalization.

    Args:
        q_identifiers_id_lev_dict (dict): a dictionary containing key-value pairs of the type (integer identifier used in nodes):(level of generalization relative to that QI)

    Returns:
        [Graph]: an instance of type graph that contains the necessary edges and nodes for the first cycle of Incognito
    """

    graph = Graph()
    # For each QI
    for q_id in dict(q_identifiers_id_lev_dict):

        # For each maximum level of the quasi identifier, setting the range between this and 0
        for level in reversed(range(q_identifiers_id_lev_dict[q_id])):

            node = Node(False, [q_id], [level])
            id_current = node.id

            # if the level is 0 there is no connection with the previous one
            if level != 0:
                graph.edges.append([id_current+1, id_current])
            if level == 0:
                node.set_is_root(True)
            graph.add_node(node)
    return graph

# -------------------------------------------------------------------


def graph_generation(s: list, edges: list):
    """
    This function generates the next graph, using the subset property, on the basis of the nodes and the edges received as input.

    Args:
        s (list): list containing the remaining nodes of the previous graph that will contribute to the formation of the nodes of the next graph
        edges (list): list of arcs of the previous graph that will contribute to the creation of the arcs of the next graph

    Returns:
        [Graph]: an instance of type graph that represents the generated graph and that will be used for the next cycle
    """
    newGraph = Graph()

    for p in range(len(s)):
        for q in range(len(s)):
            if list(s[p].q_identifiers_list)[:-1] == list(s[q].q_identifiers_list)[:-1] and \
                    list(s[p].generalization_level)[:-1] == list(s[q].generalization_level)[:-1] and \
                    s[p].q_identifiers_list[-1] < s[q].q_identifiers_list[-1]:

                qi1 = s[p].q_identifiers_list
                qi2 = s[q].q_identifiers_list
                gl1 = s[p].generalization_level
                gl2 = s[q].generalization_level

                nodeTemp = Node(False, [*qi1, qi2[-1]], [*gl1, gl2[-1]])

                nodeTemp.parent1 = s[p].id
                nodeTemp.parent2 = s[q].id
                newGraph.add_node(nodeTemp)

    candidate_edges = []
    for p in range(len(newGraph.nodes)):
        for q in range(len(newGraph.nodes)):
            for e in range(len(edges)):
                for f in range(len(edges)):
                    if (edges[e][0] == newGraph.nodes[p].parent1 and edges[e][1] == newGraph.nodes[q].parent1 and
                        edges[f][0] == newGraph.nodes[p].parent2 and edges[f][1] == newGraph.nodes[q].parent2) or \
                        (edges[e][0] == newGraph.nodes[p].parent1 and edges[e][1] == newGraph.nodes[q].parent1 and
                         newGraph.nodes[p].parent2 == newGraph.nodes[q].parent2) or \
                        (edges[e][0] == newGraph.nodes[p].parent2 and edges[e][1] == newGraph.nodes[q].parent2 and
                         newGraph.nodes[p].parent1 == newGraph.nodes[q].parent1):

                        candidate_edges.append(
                            [newGraph.nodes[p].id, newGraph.nodes[q].id])

    unique_result_edges = []

    for e in candidate_edges:
        if e not in unique_result_edges:
            unique_result_edges.append(e)

    edges_to_remove = []

    for d1 in range(len(unique_result_edges)):
        for d2 in range(len(unique_result_edges)):
            if unique_result_edges[d1][1] == unique_result_edges[d2][0]:
                edges_to_remove.append(
                    [unique_result_edges[d1][0], unique_result_edges[d2][1]])

    final_edges = []
    for e in unique_result_edges:
        if e not in edges_to_remove:
            final_edges.append(e)

    newGraph.edges = final_edges

    # Set the root nodes
    newGraph.check_roots()
    return newGraph

# -------------------------------------------------------------------


def get_frequency_set_root(df: DataFrame):
    """
    This function calculates the frequency set of a root node, using a Pandas function to count each unique tuple.

    Args:
        df (DataFrame): generalized dataset of which is needed to calculate the frequency set

    Returns:
        [DataFrame]: frequency set of the root node
    """

    qi_frequency_set = df.value_counts().reset_index(name='counts')

    return qi_frequency_set

# -------------------------------------------------------------------


def get_frequency_set(df: DataFrame, qi_attr: list):
    """
    This function calculates the frequency set of a node, using a Pandas function to group and update the count of each tuple.

    Args:
        df (DataFrame): frequency-set generalized dataset for which to update the set of frequencies
        qi_attr (list): list of QI on which to calculate the frequency set (counts excluded) (es: list of -> qi_name|lev_of_generalization)

    Returns:
        [DataFrame]: node frequency set
    """

    aggregation_functions = {'counts': 'sum'}
    qi_frequency_set = df.groupby(qi_attr).aggregate(
        aggregation_functions).reset_index()

    return qi_frequency_set

# -------------------------------------------------------------------


def check_k_anonimity(node: Node):
    """
    This function allows to check the k-anonymity from a frequency set of a particular node

    Args:
        node (Node): node instance on which to check k-anonymity

    Returns:
        [bool]: boolean value [True - if k-anonymous, False - if not k-anonymous]
    """
    is_k_anon = True
    for col, row in node.frequency_set.iteritems():
        if col == "counts":
            for el in row:
                if(el < k_anonimity):
                    is_k_anon = False
                    break
    return is_k_anon

# -------------------------------------------------------------------


def calculate_frequency_set_from_parent(frequency_set_parent: DataFrame, qi_dict_node: dict):
    """
    This function calculates the new frequency set of a node starting from the frequency set of his parent node.

    Args:
        frequency_set_parent (DataFrame): frequency set of the parent node
        qi_dict_node (dict): QI dictionary taken into account by the node, with the format (QI_name):(level of generalization relative to that QI)

    Returns:
        [DataFrame]: frequency set of the node
    """

    # Finding the list of qi with levels of generalization (es: list of -> qi_name|lev_of_generalization)
    qi_with_levels_node = []
    for key in qi_dict_node.keys():
        qi_with_levels_node.append(
            str("{}|{}").format(key, str(qi_dict_node[key])))

    # Adding the counts column to a list of attributes (es: list of -> qi_name|lev_of_generalization; + counts)
    freq_set_attr = copy.copy(qi_with_levels_node)
    freq_set_attr.append("counts")

    # Finding the QIs tag that changes from parent to child node
    old_qi = ""
    new_qi = ""
    for item in list(frequency_set_parent.columns):
        if item not in freq_set_attr:
            old_qi = item
            break
    for item in freq_set_attr:
        if item not in list(frequency_set_parent.columns):
            new_qi = item
            break

    # Shrinking the generalization table for the actual usage
    small_generalization_table = generalizations_table.filter(
        items=[old_qi, new_qi]).dropna()
    small_generalization_table = small_generalization_table.drop_duplicates()

    # 1 - standard method (finding the freq. set. from the generalized table such as for roots)
    # new_frequency_set_old = get_frequency_set_root(dataset_generalized)

    # 2 - optimized method (finding the freq. set. from the parent freq. set. joining and updating the counts)
    joined_table = pd.merge(
        frequency_set_parent, small_generalization_table, on=old_qi).filter(items=freq_set_attr)
    new_frequency_set = get_frequency_set(
        joined_table, qi_with_levels_node)

    return new_frequency_set

# -------------------------------------------------------------------


def mark_descendant(graph: Graph, node: Node):
    """
    This function marks all descendant nodes, given a certain node and a graph containing them.

    Args:
        graph (Graph): graph containing the nodes to be marked
        node (Node): node from which to calculate descendants
    """
    # Marking the node and its direct generalizations
    graph.take_node(node.id).set_marked(True)
    family = [node]
    while(len(family) != 0):
        descendant = family.pop(-1)
        for edge in graph.edges:
            if edge[0] == descendant.id:
                family.append(graph.take_node(edge[1]))
                graph.take_node(edge[1]).set_marked(True)


def core_incognito(dataset, qi_list):
    """
    Main function containing the core of the Incognito algorithm.
    https://www.researchgate.net/publication/221213050_Incognito_Efficient_Full-Domain_K-Anonymity

    Args:
        dataset (DataFrame): dataset to be k-anonymized
        qi_list (list): list containing all QIs for which it is necessary to generalize

    Returns:
        [int]: 0
    """

    graph = Graph()
    queue = []

    graph_initial = initialize_graph(q_identifiers_id_lev_dict)
    _log("[LOG] Created the initial graph")
    graph_list = [graph_initial]
    for i in range(0, len(qi_list)):
        _log("[LOG] Started the cycle %s/%s" % (i+1, len(qi_list)))

        graph = graph_list[-1]

        s = copy.copy(graph.nodes)
        roots = []

        for node in graph.nodes:
            if node.is_root == True:
                roots.append(node)

        queue.extend(roots)

        queue = sorted(queue, key=lambda n: sum(n.generalization_level))
        _log("[LOG] Analyzing the graph with a bottom-up BFS")

        while len(queue) > 0:
            node = queue.pop(0)
            if node.marked == False:
                # Generalize the dataset considering the node
                qi_dict_node2 = dict(
                    zip(node.q_identifiers_list, node.generalization_level))
                qi_dict_node = copy.copy(qi_dict_node2)
                for tag, id in q_identifiers_tag_id_dict.items():
                    for id2 in qi_dict_node2.keys():
                        if id == id2:
                            qi_dict_node[tag] = qi_dict_node.pop(id2)

                if node.is_root == True:
                    dataset_generalized = generalize_data(
                        dataset, qi_dict_node, generalizations_table)
                    _log("[LOG] Finished generalization of the root node")
                    # 1 - primo metodo (calcolo freq set)
                    new_frequency_set_root = get_frequency_set_root(
                        dataset_generalized)
                    _log("[LOG] Calculated the frequency set of the root node")

                    # 2 - metodo alternativo
                    # starting_frequency_set = copy.copy(dataset_generalized)
                    # starting_frequency_set['counts'] = 1
                    # new_frequency_set_2 = get_frequency_set(starting_frequency_set, qi_with_levels_node)

                    node.frequency_set = new_frequency_set_root

                else:

                    frequency_set_parent = graph.get_parent(node).frequency_set

                    node.frequency_set = calculate_frequency_set_from_parent(
                        frequency_set_parent, qi_dict_node)
                    _log("[LOG] Calculated the frequency set from a parent node")

                # Check k-anonimity
                is_k_anon = check_k_anonimity(node)
                if(is_k_anon):
                    _log("[LOG] This node is k-anonymous: ", endl=False)
                    node.print_info()
                    mark_descendant(graph, node)

                else:
                    s = list(filter(lambda n: n.id != node.id, s))
                    # ------------------------------------------------
                    # Insert the direct generalization to the queue

                    for edge in graph.edges:
                        if edge[0] == node.id:
                            queue.append(graph.take_node(edge[1]))
                    # ------------------------------------------------
                    # Sorting queue in height order
                    queue = sorted(queue, key=lambda node: sum(
                        node.generalization_level))

        _log("[LOG] Generating a new graph")
        g = graph_generation(s, graph.edges)
        _log("[LOG] Graph generated")
        graph_list.append(g)

    print("")
    _log("[LOG] End of the core_algorithm")
    _log("[LOG] Selecting the best generalization...")
    final_graph = graph_list[-2]
    final_graph.nodes = sorted(
        final_graph.nodes, key=lambda n: sum(n.generalization_level))
    for node in final_graph.nodes:
        if node.marked == True:
            qi_dict_node2 = dict(
                zip(node.q_identifiers_list, node.generalization_level))
            qi_dict_node = copy.copy(qi_dict_node2)
            for tag, id in q_identifiers_tag_id_dict.items():
                for id2 in qi_dict_node2.keys():
                    if id == id2:
                        qi_dict_node[tag] = qi_dict_node.pop(id2)
            dataset_generalized = generalize_data(
                dataset, qi_dict_node, generalizations_table)
            _log("[LOG] Best generalization: ", endl=False)
            node.print_info()
            for name_old in dataset_generalized:
                name_new = name_old.split("|")[0]
                dataset_generalized.rename(
                    columns={name_old: name_new}, inplace=True)
            dataset_generalized = pd.concat(
                [dataset_generalized, cutted_columns], axis=1)
            print("----------------------------------")
            print("FINAL GRAPH")
            graph.print_graph()
            print("----------------------------------")
            print("----------------------------------")
            print("GENERALIZED DATASET")
            print(dataset_generalized)
            print("----------------------------------")
            break
    return 0

# -------------------------------------------------------------------


'''
    MAIN
'''
if __name__ == "__main__":

    start_time = time.time()

    # ...................................
    # SELECTING THE DATASET

    # paper
    # dataset = pd.read_csv("datasets/paper/db_100.csv", dtype=str)
    # dataset = pd.read_csv("datasets/paper/db_10000.csv", dtype=str)
    # dataset = dataset.drop(["id", "disease"], axis=1)

    # adult
    dataset = pd.read_csv("datasets/adult/adult.csv", dtype=str, sep=(";"))
    dataset = dataset.loc[:265, :]
    """ cutted_columns = dataset.loc[:, ["race", "marital-status",
                                     "workclass", "occupation", "salary-class"]]
    dataset = dataset.drop(["ID", "race", "marital-status",
                            "workclass", "occupation", "salary-class"], axis=1) """
    cutted_columns = dataset.loc[:, ["workclass", "salary-class"]]
    dataset = dataset.drop(["ID", "workclass", "salary-class"], axis=1)

    _log("[LOG] Dataset loaded")
    print(dataset)

    # ...................................
    # DEFINING INPUTS

    # K-anonimity
    k_anonimity = 2
    _log("[LOG] Started with k-anonimity: %s" % k_anonimity)

    # adult
    """ q_identifiers_list_string = ["sex", "age", "education", "native-country"]
    q_identifiers_list = [1, 2, 3, 4]
    generalization_levels = [2, 5, 4, 3] """
    q_identifiers_list_string = ["sex", "age", "education", "native-country", "marital-status", "occupation", "race"]
    q_identifiers_list = [1, 2, 3, 4, 5, 6, 7]
    generalization_levels = [2, 5, 4, 3, 3, 3, 2]

    _log("[LOG] Quasi-identifier to anonymize: %s" % q_identifiers_list_string)

    # datafly
    # q_identifiers_list_string = ["age", "city_birth", "zip_code"]
    # generalization_levels = [4, 4, 6]

    # paper
    # q_identifiers_list_string = ["birthdate", "sex", "zip_code"]
    # generalization_levels = [2, 2, 3]

    # ...................................

    # ............. Plot .................
    for attr in q_identifiers_list_string:
        sns.displot(dataset, x=attr)
    plt.show()
    _log("[LOG] Plotted the distribution of each QI")

    # PREPARATION OF VARIABLES AND STRUCTURES

    # getting the QI dictionaries
    q_identifiers_tag_id_dict = dict(
        zip(q_identifiers_list_string, q_identifiers_list))
    q_identifiers_id_lev_dict = dict(
        zip(q_identifiers_list, generalization_levels))

    # getting the generalization table
    generalizations_table = create_generalization_hierarchies(
        q_identifiers_list_string, q_identifiers_tag_id_dict, q_identifiers_id_lev_dict)

    # ...................................
    # INCOGNITO ALGORITHM

    # calling the Incognito algorithm
    core_incognito(dataset, q_identifiers_list)

    # ...................................
    # EVALUATION OF THE RESULTS

    # showing evaluation parameters
    _log("[LOG] Execution time: %s s" % str(time.time() - start_time))

    # ...................................
