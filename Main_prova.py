import pandas as pd
import copy
import time
import collections
import Frequency_list_attempt

from pandas.core.frame import DataFrame
from Graph import Graph
from Node import Node


start_time = time.time()
df = pd.read_csv("datasets/db_10000.csv", dtype=str)
df = df.drop(["id","disease"],1)

q_identifiers_list = [1, 2, 3]
generalization_levels = [4, 4, 6]
q_identifiers_dict = dict(zip(q_identifiers_list, generalization_levels))
q_identifiers_tag_id_dict = {"age" : 1, "city_birth" : 2 , "zip_code" : 3}

gen = {"zip_code" : 3, "age" : 2, "city_birth" : 2}
K_anonimity = 2
graph = Graph()
# counter rappresenta l'id
# Il primo vettore identifica quali q identifiers vengono presi, mentre il secondo è il vettore di livelli di generalizzazione
a = Node(False, [1, 2], [0,0])
b = Node(False, [1, 2], [1,1])
c = Node(False, [1, 2], [0,0])
d = Node(False, [1, 2], [1,2])
e = Node(False, [3, 2], [1,0])
f = Node(False, [3, 2], [1,1])
g = Node(False, [3, 2], [0,2])
h = Node(False, [3, 2], [1,2])
i = Node(False, [3, 1], [1,0])
l = Node(False, [3, 1], [0,1])
m = Node(False, [3, 1], [1,1])

graph.add_node(a)
graph.add_node(b)
graph.add_node(c)
graph.add_node(d)
graph.add_node(e)
graph.add_node(f)
graph.add_node(g)
graph.add_node(h)
graph.add_node(i)
graph.add_node(l)
graph.add_node(m)

graph.edges=[[0,1],
            [1,3],
            [2,3],
            [4,5],
            [5,7],
            [6,7],
            [8,10],
            [9,10]]

ce = []
ce.append(a)
ce.append(c)

for el in ce:
    print(el.id)
#Sorting dei nodi usando come altezza a somma dei livelli di generalizzazione
ce = sorted(ce, key= lambda node: sum(node.generalization_level))

for el in ce:
    print(el.id)




'''
    Questa funzione serve per inizializzare il primo grafo. Verranno creani per ogni quasi identifiers n nodi
    dove n é il numero totale di generalizzazioni per quel quasi identifier
    es: sex con 2 generalizzazioni (0,1), zipcode con 5 generalizzazioni (0,1,2,3,4) --> (sex,0)(sex,1) con arco (0,1)
    (zip 0)(zip 1)...(zip 5) con arco tra (2,3)(3,4)(...) (i nodi hanno id progressivo da 1)
'''
def initialize_graph(q_identifiers_dict):
    #print(q_identifiers_dict)
    graph = Graph()
    #Per ogni quasi identifier
    for q_id in dict(q_identifiers_dict):
        
        #per ogni livello massimo del quasi identifier creo il range tra questo e 0
        for level in reversed(range(q_identifiers_dict[q_id])):
            
            node = Node(False, q_id, level)
            id_current = node.id

            #se il livello é 0 non c'é un collegamento con il precendente
            if level !=0:
                graph.edges.append([id_current+1,id_current])
            if level == 0:
                node.set_is_root(True)
            graph.add_node(node)
    return graph


def graph_generation(s:list, edges:list):

    newGraph = Graph()


    for p in range(len(s)):
        for q in range(len(s)):
            if list(s[p].q_identifier_list)[:-1]==list(s[q].q_identifier_list)[:-1] and \
                list(s[p].generalization_level)[:-1]==list(s[q].generalization_level)[:-1] and \
                s[p].q_identifier_list[-1] < s[q].q_identifier_list[-1]:
                
                qi1=s[p].q_identifier_list
                qi2=s[q].q_identifier_list
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
    
    #newGraph.print_graph()

'''
    Questa funziona genera una tabella con tutte le possibili generalizzazioni per ogni QI, crea un dizionario di dizionari
    con come chiave il tag del QI, mentre come valore un dizionario che a sua volta ha come chiave il livello di generalizzazione e come 
    valore la lista di tutte le generalizzazioni di quel livello
'''
""" def create_generalization_hierarchies(generalization_level:dict):
    all_gen = collections.defaultdict(dict)
    for tag, level in generalization_level.items():
        path = str("datasets/{}_generalization.csv").format(tag)
        df_all_gen = pd.read_csv(path, header=None, dtype=str)
        for key, qi_id in q_identifiers_tag_id_dict.items():
            if key == tag:
                for i in range(0, q_identifiers_dict[qi_id]):
                    all_gen[tag][i] = df_all_gen.iloc[:,i]
        
    return all_gen """

'''
    Questa funziona genera una tabella con tutte le possibili generalizzazioni per ogni QI, crea un dizionario di dizionari
    con come chiave il tag del QI, mentre come valore un dizionario che a sua volta ha come chiave il livello di generalizzazione e come 
    valore la lista di tutte le generalizzazioni di quel livello
'''
def create_generalization_hierarchies(generalization_level:dict):
    all_gen = pd.DataFrame()
    for tag, level in generalization_level.items():
        path = str("datasets/{}_generalization.csv").format(tag)
        df_all_gen = pd.read_csv(path, header=None, dtype=str)
        for key, qi_id in q_identifiers_tag_id_dict.items():
            if key == tag:
                for i in range(0, q_identifiers_dict[qi_id]):
                    column_name = str("{}{}").format(key,i)
                    all_gen[column_name] = df_all_gen.iloc[:,i]
        
    return all_gen
        
""" '''
    Questa funzione prende in input il df, le generalizzazioni richieste e un dizionario contenente tutte le generalizzazioni per ogni QI.
    Cicla su tutte le coppie chiavi-valore presenti nel dizionario delle generalizzazioni richieste e in base al livello di generalizzazione
    richiesto prende dalla tabella contenente tutte le gen. la prima colonna(valore originale) e la colonna del livello richiesto.
    A questo punto sostituisce con il valore anonimizzato
'''
def generalize_data(df:DataFrame, generalization_levels:dict, all_generalizations:DataFrame):
    df_gen = copy.copy(df)
    for index, level in generalization_levels.items():
        to_generalize = df.loc[:, index]
        lookup = dict(zip(all_generalizations[index][0], all_generalizations[index][level]))
        for row in to_generalize:
            for original, anonymized in lookup.items():
                if str(original) == str(row):
                    to_generalize.replace(to_replace = str(row), inplace=True, value = str(anonymized))
        df[index] = to_generalize
        
    return df_gen
"""

'''
    Questa funzione prende in input il df, le generalizzazioni richieste e un dizionario contenente tutte le generalizzazioni per ogni QI.
    Cicla su tutte le coppie chiavi-valore presenti nel dizionario delle generalizzazioni richieste e in base al livello di generalizzazione
    richiesto prende dalla tabella contenente tutte le gen. la prima colonna(valore originale) e la colonna del livello richiesto.
    A questo punto sostituisce con il valore anonimizzato
'''
def generalize_data(df:DataFrame, generalization_levels:dict, all_generalizations:DataFrame):
    df_gen = copy.copy(df)
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

""" 
def core_incognito(graph:Graph):

    queue = []
    
    for i in range(0, len(graph.q_identifier_list)):
        s = copy.copy(graph.nodes)
        roots = []
        #TODO roots in graph_generation
        #TODO sort the nodes on the deep of the nodes - Alessio
        for node in graph.nodes:
            if node.is_root == True:
                roots.append(node)
        queue.extend(roots)

        while queue != False:
            node = queue.pop()
            if node.marked == False:
                q_id_dict = dict(zip(node.q_identifier_list, node.generalization_level))
                df_gen = generalize_data(df, q_id_dict, all_generalizations)
                if node.is_root == True:
                    # TODO: Calcolare frequency set dalla tabella
                    frequency_set = Frequency_list_attempt.get_frequency_list_pandas(df_gen, node.q_identifiers_list)
                else:
                    # TODO: Calcolare frequency set dai parenti del nodo

                    new_df = df_gen.set_index('key').join(frequency_set_parent('key'))
                    
                    frequency_set = Frequency_list_attempt.get_frequency_list_pandas(df_gen, node.q_identifiers_list)
            # TODO: Controllare la k-anonimity partendo dal frequency set
            # Da definire la k-anonymity
            k = 0
            if(k>=K_anonimity):
                # marka il nodo stesso e i nodi vicini 
                graph.take_node(node.id).set_marked(True)
                #s.take_node(node.id).set_marked(True) # NON VA, s é una lista non un grafo
                for edge in graph.edges:
                    if edge[0]==node.id:
                        graph.take_node(edge[1]).set_marked(True)
            else:
                # ci vuole l'indice non node
                # TODO: cancellare il nodo stesso -> da provare
                s = filter(lambda n: n.id != node.id, s)
                # TODO: inserire le dirette generalizzazioni del nodo nella coda
                for edge in graph.edges:
                    if edge[0]==node.id:
                        queue.append(graph.take_node(edge[1]))
        # TODO: Graph Generation per passare al grafo successivo (da passare le due liste, nodi e grafi)
        graph2 = graph_generation(s, graph.edges)


core_incognito(graph) """

#all_generalizations = create_generalization_hierarchies(gen)
#df_gen = generalize_data(df, gen, all_generalizations)
#print("Execution time: "+str(time.time() - start_time)+"s")