from numpy import pi
import pandas as pd
import copy
import time
import collections

from pandas.core.frame import DataFrame
from Graph import Graph
from Node import Node

#-------------------------------------------------------------------

def initialize_graph(q_identifiers_tag_lev_dict):
    
    graph = Graph()
    #Per ogni quasi identifier
    for q_id in dict(q_identifiers_tag_lev_dict):
        
        #per ogni livello massimo del quasi identifier creo il range tra questo e 0
        for level in reversed(range(q_identifiers_tag_lev_dict[q_id])):
            
            node = Node(False, [q_id], [level])
            id_current = node.id

            #se il livello é 0 non c'é un collegamento con il precendente
            if level !=0:
                graph.edges.append([id_current+1,id_current])
            if level == 0:
                node.set_is_root(True)
            graph.add_node(node)
    return graph

#-------------------------------------------------------------------

def graph_generation(s:list, edges:list):

    newGraph = Graph()


    for p in range(len(s)):
        for q in range(len(s)):
            if list(s[p].q_identifiers_list)[:-1]==list(s[q].q_identifiers_list)[:-1] and \
                list(s[p].generalization_level)[:-1]==list(s[q].generalization_level)[:-1] and \
                s[p].q_identifiers_list[-1] < s[q].q_identifiers_list[-1]:
                
                qi1=s[p].q_identifiers_list
                qi2=s[q].q_identifiers_list
                gl1=s[p].generalization_level
                gl2=s[q].generalization_level
                
               
                nodeTemp = Node(False, [*qi1, qi2[-1]], [*gl1, gl2[-1]])
                
                
                nodeTemp.parent1 = s[p].id
                nodeTemp.parent2 = s[q].id
                newGraph.add_node(nodeTemp)
        
    candidate_edges=[]
    for p in range(len(newGraph.nodes)):
        for q in range(len(newGraph.nodes)):
            for e in range(len(edges)):
                for f in range(len(edges)):
                    if (edges[e][0]==newGraph.nodes[p].parent1 and edges[e][1]==newGraph.nodes[q].parent1 and \
                        edges[f][0]==newGraph.nodes[p].parent2 and edges[f][1]==newGraph.nodes[q].parent2) or \
                        (edges[e][0]==newGraph.nodes[p].parent1 and edges[e][1]==newGraph.nodes[q].parent1 and \
                        newGraph.nodes[p].parent2 == newGraph.nodes[q].parent2) or \
                        (edges[e][0]==newGraph.nodes[p].parent2 and edges[e][1]==newGraph.nodes[q].parent2 and \
                         newGraph.nodes[p].parent1 == newGraph.nodes[q].parent1):

                            candidate_edges.append([newGraph.nodes[p].id, newGraph.nodes[q].id])
    
    

    unique_result_edges = []
    
    for e in candidate_edges:
        if e not in unique_result_edges:
            unique_result_edges.append(e)


    edges_to_remove=[]
    
    for d1 in range(len(unique_result_edges)):
        for d2 in range(len(unique_result_edges)): 
            #print(str(d1) + str(d2))
            if unique_result_edges[d1][1]==unique_result_edges[d2][0]:
                edges_to_remove.append([unique_result_edges[d1][0],unique_result_edges[d2][1]])
    
    final_edges = []
    for e in unique_result_edges:
        if e not in edges_to_remove:
            final_edges.append(e)
    
    newGraph.edges=final_edges

    #Setta i nodi roots
    newGraph.check_roots()

#-------------------------------------------------------------------

'''
    Questa funziona genera una tabella con tutte le possibili generalizzazioni per ogni QI, crea un dizionario di dizionari
    con come chiave il tag del QI, mentre come valore un dizionario che a sua volta ha come chiave il livello di generalizzazione e come 
    valore la lista di tutte le generalizzazioni di quel livello
'''
def create_generalization_hierarchies(generalization_level):
    all_gen = pd.DataFrame()
    for tag in generalization_level:
        path = str("datasets/{}_generalization.csv").format(str(tag))
        df_all_gen = pd.read_csv(path, header=None, dtype=str)
        for key, qi_id in q_identifiers_tag_id_dict.items():
            if key == tag:
                for i in range(0, q_identifiers_tag_lev_dict[qi_id]):
                    column_name = str("{}{}").format(key,i)
                    all_gen[column_name] = df_all_gen.iloc[:,i]
        
    return all_gen

#-------------------------------------------------------------------

'''
    Questa funzione prende in input il df, le generalizzazioni richieste e un dizionario contenente tutte le generalizzazioni per ogni QI.
    Cicla su tutte le coppie chiavi-valore presenti nel dizionario delle generalizzazioni richieste e in base al livello di generalizzazione
    richiesto prende dalla tabella contenente tutte le gen. la prima colonna(valore originale) e la colonna del livello richiesto.
    A questo punto sostituisce con il valore anonimizzato
'''
def generalize_data(dataset:DataFrame, generalization_levels:dict, all_generalizations:DataFrame):
    df_gen = copy.copy(dataset)
    for index, level in generalization_levels.items():
        to_generalize = df_gen.loc[:, index]
        lev_string = str("{}{}").format(index, level)
        ind_string = str("{}{}").format(index, 0)
        lookup = pd.Series(all_generalizations[lev_string].values, index=all_generalizations[ind_string]).to_dict()
        for row in to_generalize:
            for original, anonymized in lookup.items():
                if str(original) == str(row):
                    to_generalize.replace(to_replace = str(row), inplace=True, value = str(anonymized))
        df_gen[index] = to_generalize
        
    return df_gen

#-------------------------------------------------------------------

'''
    FREQUENCY LIST CON PANDAS
    Questa funzione permette di ottenere la frequency list del dataset rispetto agli attributi qi indicati.
    
    :param dataframe contenente la tabella
    :param lista di qi da considerare nel conteggio per la frequency list

    :return frequency_list and number of unique elements
'''
def get_frequency_list_pandas(df:DataFrame, qi_list:list):      # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html
    
    # tuple -> (counts, {row_keys})
    qi_frequency = dict()
    
    for tag, id in q_identifiers_tag_id_dict.items():
        for index, id2 in enumerate(qi_list):
            if id == id2:
                qi_list[index] = tag

    qi_frequency = df[qi_list].value_counts().rename_axis(qi_list).reset_index(name='counts')
    #qi_frequency = qi_frequency.to_dict()
    #qi_frequency = qi_frequency.to_frame()

    counter = len(qi_frequency)
    return qi_frequency, counter

#-------------------------------------------------------------------
start_time = time.time()
dataset = pd.read_csv("datasets/db_100.csv", dtype=str)
dataset = dataset.drop(["id","disease"], axis=1)

#-------------------------------------------------------------------

# INPUTS
k_anonimity = 2
q_identifiers_list = [1, 2, 3]
q_identifiers_list_string = ["age","city_birth","zip_code"]
generalization_levels = [4, 4, 6]   # anche ottenibile da file

q_identifiers_tag_id_dict = {"age" : 1, "city_birth" : 2 , "zip_code" : 3}

q_identifiers_tag_lev_dict = dict(zip(q_identifiers_list, generalization_levels))
generalizations_table = create_generalization_hierarchies(q_identifiers_list_string)

#-------------------------------------------------------------------



def core_incognito(dataset, qi_list):

   
    graph = Graph()
    queue = []
    
    graph_initial = initialize_graph(q_identifiers_tag_lev_dict)

    for i in range(0, len(qi_list)):
        print("ciclo iniziato") 
        graph_list = [graph_initial]
        
        graph=graph_list[-1]

        s = copy.copy(graph.nodes)
        roots = []
        
        for node in graph.nodes:
            if node.is_root == True:
                roots.append(node)
        
        queue.extend(roots)

        print(sum(queue[1].generalization_level))
        
        queue = sorted(queue, key = lambda n: sum(n.generalization_level))

        while queue != False:
            node = queue.pop()
            if node.marked == False:
                #Generalizzare il dataset considerando il nodo
                qi_dict_node2 = dict(zip(node.q_identifiers_list, node.generalization_level))
                qi_dict_node = copy.copy(qi_dict_node2)
                for tag, id in q_identifiers_tag_id_dict.items():
                    for id2 in qi_dict_node2.keys():
                        if id == id2:
                            qi_dict_node[tag] = qi_dict_node.pop(id2)
                print(qi_dict_node)
                dataset_generalized = generalize_data(dataset, qi_dict_node, generalizations_table)

                if node.is_root == True:
                    node.frequency_set = get_frequency_list_pandas(dataset_generalized, node.q_identifiers_list)
                else:
                    #generalized_frequency_set = frequency_set_parent.join(generalizations_table, lsuffix='', rsuffix='0')
                    #node.frequency_set = get_frequency_list_pandas(generalized_frequency_set, node.q_identifiers_list)
                    #
                    node.frequency_set = get_frequency_list_pandas(dataset_generalized, node.q_identifiers_list)

                #---------------------
                is_k_anon = False
                for row in node.frequency_set:
                    print(row["counts"])
                    if((row["counts"] >= k_anonimity).any()):
                        is_k_anon = True
                        break
                #--------------------- 
                if(is_k_anon):
                    #------------------------------------------------
                    #Marca il nodi e le suoi dirette generalizzazioni 
                    graph.take_node(node.id).set_marked(True)
                    for edge in graph.edges:
                        if edge[0]==node.id:
                            graph.take_node(edge[1]).set_marked(True)       
                    #------------------------------------------------ 
                else:
                    s = filter(lambda n: n.id != node.id, s)
                    #------------------------------------------------
                    #Inserire dirette generazioni nella coda
                    for edge in graph.edges:
                        if edge[0]==node.id:
                            queue.append(graph.take_node(edge[1]))
                    #------------------------------------------------
                    queue = sorted(queue, key= lambda node: sum(node.generalization_level)) # TODO DA CONTROLLARE SE FUNZIONA SE É ULTIMO NODO

        print("ciclo terminato")        
        graph.print_graph()
        # TODO: Graph Generation per passare al grafo successivo (da passare le due liste, nodi e grafi)
        graph_list.append(graph_generation(s, graph.edges))

        print("finito")

    return 0
    
#-------------------------------------------------------------------

core_incognito(dataset, q_identifiers_list)