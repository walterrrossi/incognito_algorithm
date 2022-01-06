from numpy import pi
import pandas as pd
import copy
import time

from pandas.core.frame import DataFrame
from generalization import create_generalization_hierarchies
from generalization import generalize_data
from Graph import Graph
from Node import Node

# -------------------------------------------------------------------

'''
    Questa funzione inizializza il primo grafo dell'algoritmo. Questo grafo dovrá contenere come nodi i quasi identifier singoli con i loro
    livelli di generalizzazione.

    :param q_identifiers_id_lev_dict è un dizionario contenente coppie chiavi-valore del tipo (identificativo intero usato nei nodi):(livello di generalizzazione relativo a quel QI)
    :return graph un'istanza di tipo grapo che contiene gli archi e nodi necessari per il primo ciclo dell'incognito
'''
def initialize_graph(q_identifiers_id_lev_dict):

    graph = Graph()
    # Per ogni quasi identifier
    for q_id in dict(q_identifiers_id_lev_dict):

        # per ogni livello massimo del quasi identifier creo il range tra questo e 0
        for level in reversed(range(q_identifiers_id_lev_dict[q_id])):

            node = Node(False, [q_id], [level])
            id_current = node.id

            # se il livello é 0 non c'é un collegamento con il precendente
            if level != 0:
                graph.edges.append([id_current+1, id_current])
            if level == 0:
                node.set_is_root(True)
            graph.add_node(node)
    return graph

# -------------------------------------------------------------------

'''
    Questa funziona genera una tabella con tutte le possibili generalizzazioni per ogni QI, crea un dizionario di dizionari
    con come chiave il tag del QI, mentre come valore un dizionario che a sua volta ha come chiave il livello di generalizzazione e come 
    valore la lista di tutte le generalizzazioni di quel livello

    :param s è una lista contenente i nodi rimasti del grafo precendente che contribuiranno alla formazione dei nodi 
            del grafo sucessivo
    :param edges è la lista di archi del grafo precendente che contribuirá alla creazione degli archi del grafo sucessivo
    :return newGraph è un'istanza di tipo grafo che rappresenta il grafo generato e che sará utilizzato per il ciclo sucessivo
'''
def graph_generation(s: list, edges: list):

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

    # Setta i nodi roots
    newGraph.check_roots()
    return newGraph

# -------------------------------------------------------------------


'''
    FREQUENCY LIST CON PANDAS
    Questa funzione permette di ottenere la frequency list del dataset rispetto agli attributi qi indicati.
    
    :param dataframe contenente la tabella
    :param lista di qi da considerare nel conteggio per la frequency list

    :return frequency_list and number of unique elements
'''


# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html
def get_frequency_list_pandas(df: DataFrame, qi_list: list):

    # tuple -> (counts, {row_keys})
    qi_frequency = dict()
    qi_list_copy = copy.copy(qi_list)
    for tag, id in q_identifiers_tag_id_dict.items():
        for index, id2 in enumerate(qi_list):
            if id == id2:
                qi_list_copy[index] = tag

    qi_frequency = df[qi_list_copy].value_counts().rename_axis(
        qi_list_copy).reset_index(name='counts')
    #qi_frequency = qi_frequency.to_dict()
    #qi_frequency = qi_frequency.to_frame()

    return qi_frequency

# -------------------------------------------------------------------

'''
    Questa funzione permette di verificare la k-anonimity su una frequency list contenuta da un nodo
    
    :param node istanza di tipo node
    :return is_k_anon nu valore booleano True se k-anonimus o viceversa
'''

def check_k_anonimity(node: Node):
    is_k_anon = True
    for col, row in node.frequency_set.iteritems():
        if col == "counts":
            for el in row:
                if(el < k_anonimity):
                    is_k_anon = False
                    break
    return is_k_anon

# -------------------------------------------------------------------


def core_incognito(dataset, qi_list):

    graph = Graph()
    queue = []

    graph_initial = initialize_graph(q_identifiers_id_lev_dict)
    graph_list = [graph_initial]
    for i in range(0, len(qi_list)):
        print("ciclo iniziato")

        graph = graph_list[-1]

        s = copy.copy(graph.nodes)
        roots = []

        for node in graph.nodes:
            if node.is_root == True:
                roots.append(node)

        queue.extend(roots)

        queue = sorted(queue, key=lambda n: sum(n.generalization_level))

        while len(queue) > 0:
            node = queue.pop(0)
            if node.marked == False:
                # Generalizzare il dataset considerando il nodo
                qi_dict_node2 = dict(
                    zip(node.q_identifiers_list, node.generalization_level))
                qi_dict_node = copy.copy(qi_dict_node2)
                for tag, id in q_identifiers_tag_id_dict.items():
                    for id2 in qi_dict_node2.keys():
                        if id == id2:
                            qi_dict_node[tag] = qi_dict_node.pop(id2)
                dataset_generalized = generalize_data(
                    dataset, qi_dict_node, generalizations_table)
                if node.is_root == True:
                    node.frequency_set = get_frequency_list_pandas(
                        dataset_generalized, node.q_identifiers_list)
                else:
                    #generalized_frequency_set = frequency_set_parent.join(generalizations_table, lsuffix='', rsuffix='0')
                    #node.frequency_set = get_frequency_list_pandas(generalized_frequency_set, node.q_identifiers_list)
                    #
                    node.frequency_set = get_frequency_list_pandas(
                        dataset_generalized, node.q_identifiers_list)

                is_k_anon = check_k_anonimity(node)
                if(is_k_anon):
                    # ------------------------------------------------
                    # Marca il nodi e le suoi dirette generalizzazioni
                    graph.take_node(node.id).set_marked(True)
                    for edge in graph.edges:
                        if edge[0] == node.id:
                            graph.take_node(edge[1]).set_marked(True)
                    # ------------------------------------------------
                else:
                    s = list(filter(lambda n: n.id != node.id, s))
                    # ------------------------------------------------
                    # Inserire dirette generazioni nella coda
                    for edge in graph.edges:
                        if edge[0] == node.id:
                            queue.append(graph.take_node(edge[1]))
                    # ------------------------------------------------
                    # Sort della lista in ordine di altezza
                    queue = sorted(queue, key=lambda node: sum(
                        node.generalization_level))

        graph.print_graph()

        print("ciclo terminato")

        print("S:")
        for m in s:
            m.print_info()
        print("Grafo:")
        g = graph_generation(s, graph.edges)
        g.print_graph()
        graph_list.append(g)

    print("finito")
    final_graph = graph_list[-2]
    final_graph.nodes = sorted(
        final_graph.nodes, key=lambda n: sum(n.generalization_level))
    for node in final_graph.nodes:
        if node.marked == True:
            node.print_info()
            qi_dict_node2 = dict(
                zip(node.q_identifiers_list, node.generalization_level))
            qi_dict_node = copy.copy(qi_dict_node2)
            for tag, id in q_identifiers_tag_id_dict.items():
                for id2 in qi_dict_node2.keys():
                    if id == id2:
                        qi_dict_node[tag] = qi_dict_node.pop(id2)
            dataset_generalized = generalize_data(
                dataset, qi_dict_node, generalizations_table)
            print(dataset_generalized)
            break
    return 0

# -------------------------------------------------------------------


# DATASET
start_time = time.time()
#dataset = pd.read_csv("datasets/db_100.csv", dtype=str)

dataset = pd.read_csv("datasets/db_100.csv", dtype=str)
dataset = dataset.drop(["id", "disease"], axis=1)

# -------------------------------------------------------------------


# INPUTS
k_anonimity = 2
q_identifiers_list = [1, 2, 3]
q_identifiers_list_string = ["age", "city_birth", "zip_code"]
#q_identifiers_list_string = ["birthdate", "sex", "zip_code"]
generalization_levels = [4, 4, 6]   # anche ottenibile da file
# generalization_levels = [2, 2, 3]   # anche ottenibile da file

q_identifiers_tag_id_dict = dict(
    zip(q_identifiers_list_string, q_identifiers_list))
#q_identifiers_tag_id_dict = dict(zip(q_identifiers_list_string, q_identifiers_list))

q_identifiers_id_lev_dict = dict(
    zip(q_identifiers_list, generalization_levels))
generalizations_table = create_generalization_hierarchies(
    q_identifiers_list_string, q_identifiers_tag_id_dict, q_identifiers_id_lev_dict)
print(generalizations_table)

core_incognito(dataset, q_identifiers_list)

print("Execution time: " + str(time.time() - start_time) + "s")
