import pandas as pd
import copy
import time

from pandas.core.frame import DataFrame
from Graph import Graph
from Node import Node


start_time = time.time()
df = pd.read_csv("datasets/db_10000.csv", dtype=str)
df = df.drop(["id","disease"],1)
#print(df)

q_identifiersList = [1, 2, 3]
generalization_levels = [2,6,4]
q_identifiersDict = {1:2, 2:6, 3:4}

gen = dict()
gen["zip_code"] = 3
gen["age"] = 2
gen["city_birth"] = 2
K_anonimity = 2
graph = Graph()
# counter rappresenta l'id
# Il primo vettore identifica quali q identifiers vengono presi, mentre il secondo è il vettore di livelli di generalizzazione
a = Node(False, [1, 2], [1,0])

b = Node(False, [1, 2], [1,1])

c = Node(False, [1, 2], [0,2])

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

graph.edges=[[1,2],
            [2,4],
            [3,4],
            [5,6],
            [6,8],
            [7,8],
            [9,11],
            [10,11]]


   
'''
def core_incognito(graph:Graph):

    queue = []
    
    for i in range(0, len(graph.q_identifier_list)):
        s = dict()
        s = copy.copy(graph.nodes)
        roots = list()
        for node in graph.nodes.values():
            if node.is_root == True:
                roots.append(node)
        queue.extend(roots)
    print(queue)

    while queue != False:
        node = queue.pop()
        if node.marked == False:
            if node.is_root == True:
                # TODO: Calcolare frequency set dalla tabella
                frequency_set
            else:
                # TODO: Calcolare frequency set dai parenti del nodo
                frequency_set
        # TODO: Controllare la k-anonimity partendo dal frequency set
        # Da definire la k-anonymity
        k = 0
        if(k>=K_anonimity):
            # TODO: markare il nodo stesso e i nodi vicini
            pass
        else:
            # ci vuole l'indice non node
            # TODO: cancellare il nodo stesso (anche edge collegati)
            # TODO: inserire le dirette generalizzazioni del nodo nella coda       
            pass
    # TODO: Graph Generation per passare al grafo successivo


for qi in qi_list:
    graph = Graph()
    graph.initialize(qi, gen)

core_incognito(graph)

'''


def initialize_graph(q_identifiersDict):
    print(q_identifiersDict)
    graph = Graph()
    
    for q_id in dict(q_identifiersDict):
        print(q_id)
        for level in reversed(range(q_identifiersDict[q_id])):
            print(level)
            node = Node(False, q_id, level)
            id_current = node.id
            if level !=0:
                graph.edges.append([id_current+1,id_current])
            if level == 0:
                node.set_is_root(True)
            graph.add_node(node)
    return graph

graph.print_graph()

prova_primo_graph = Graph()
prova_primo_graph = initialize_graph(q_identifiersDict)

prova_primo_graph.print_graph()



def graph_generation(counter, graph:Graph):

    newGraph = Graph()


    for p in range(len(graph.nodes)):
        for q in range(len(graph.nodes)):
            if list(graph.nodes[p].q_identifier_list)[:-1]==list(graph.nodes[q].q_identifier_list)[:-1] and \
                list(graph.nodes[p].generalization_level)[:-1]==list(graph.nodes[q].generalization_level)[:-1] and \
                graph.nodes[p].q_identifier_list[-1] < graph.nodes[q].q_identifier_list[-1]:
                
                qi1=graph.nodes[p].q_identifier_list
                qi2=graph.nodes[q].q_identifier_list
                gl1=graph.nodes[p].generalization_level
                gl2=graph.nodes[q].generalization_level
                
               
                nodeTemp = Node(counter, False, [*qi1, qi2[-1]], [*gl1, gl2[-1]])
                counter+=1
                
                nodeTemp.parent1 = graph.nodes[p].id
                nodeTemp.parent2 = graph.nodes[q].id
                newGraph.add_node(nodeTemp)
        
    candidate_edges=[]
    for p in range(len(newGraph.nodes)):
        for q in range(len(newGraph.nodes)):
            for e in range(len(graph.edges)):
                for f in range(len(graph.edges)):
                    if (graph.edges[e][0]==newGraph.nodes[p].parent1 and graph.edges[e][1]==newGraph.nodes[q].parent1 and \
                        graph.edges[f][0]==newGraph.nodes[p].parent2 and graph.edges[f][1]==newGraph.nodes[q].parent2) or \
                        (graph.edges[e][0]==newGraph.nodes[p].parent1 and graph.edges[e][1]==newGraph.nodes[q].parent1 and \
                        newGraph.nodes[p].parent2 == newGraph.nodes[q].parent2) or \
                        (graph.edges[e][0]==newGraph.nodes[p].parent2 and graph.edges[e][1]==newGraph.nodes[q].parent2 and \
                         newGraph.nodes[p].parent1 == newGraph.nodes[q].parent1):

                            candidate_edges.append([newGraph.nodes[p].id, newGraph.nodes[q].id])
    
    
    #newGraph.print_nodes()
    

    unique_result_edges = []
    
    for e in candidate_edges:
        if e not in unique_result_edges:
            unique_result_edges.append(e)

    #print(unique_result_edges)

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
    
    #print(unique_result_edges)
    #print(final_edges)

    newGraph.edges=final_edges

    newGraph.print_graph()


def create_generalization_hierarchies(id:str):
    all_gen = dict()
    if id == "zip_code":
        df_all_gen = pd.read_csv("datasets/zip_code_generalization.csv", header=None, dtype=str)
        all_gen["0"] = df_all_gen.iloc[:,0]
        all_gen["1"] = df_all_gen.iloc[:,1]
        all_gen["2"] = df_all_gen.iloc[:,2]
        all_gen["3"] = df_all_gen.iloc[:,3]
        all_gen["4"] = df_all_gen.iloc[:,4]
        all_gen["5"] = df_all_gen.iloc[:,5]
    if id == "age":
        df_all_gen = pd.read_csv("datasets/age_generalization.csv", header=None, dtype=str)
        all_gen["0"] = df_all_gen.iloc[:,0]
        all_gen["1"] = df_all_gen.iloc[:,1]
        all_gen["2"] = df_all_gen.iloc[:,2]
        all_gen["3"] = df_all_gen.iloc[:,3]
    if id == "city_birth":
        df_all_gen = pd.read_csv("datasets/city_birth_generalization.csv", header=None, dtype=str)
        all_gen["0"] = df_all_gen.iloc[:,0]
        all_gen["1"] = df_all_gen.iloc[:,1]
        all_gen["2"] = df_all_gen.iloc[:,2]
        all_gen["3"] = df_all_gen.iloc[:,3]
    return all_gen
        

# Per ora generalizzo in forza bruta, prendo il livello di generalizzazione e 
# sostituisco con la colonna corrispondente del csv delle generalizzazioni
def generalize_data(df:DataFrame, generalization_level:dict):
    for index, level in generalization_level.items():
        if index == "zip_code":
            possible_generalizations = pd.read_csv("datasets/zip_code_generalization.csv", header=None, dtype=str)
            to_generalize = df.loc[:, index]
            from_generalize = possible_generalizations.iloc[:,[0,level]]
            for row in to_generalize:
                for i, pattern in from_generalize.iterrows():
                    if  str(pattern.iloc[0]) == str(row):
                        to_generalize.replace(to_replace = str(row), inplace=True, value = str(pattern.iloc[1]))
            df[index] = to_generalize

        if index == "age":
            possible_generalizations = pd.read_csv("datasets/age_generalization.csv", header=None, dtype=str)
            to_generalize = df.loc[:, index]
            from_generalize = possible_generalizations.iloc[:,[0,level]]
            for row in to_generalize:
                for i, pattern in from_generalize.iterrows():
                    if  str(pattern.iloc[0]) == str(row):
                        to_generalize.replace(to_replace = str(row), inplace=True, value = str(pattern.iloc[1]))
            df[index] = to_generalize
        
        if index == "city_birth":
            possible_generalizations = pd.read_csv("datasets/city_birth_generalization.csv", header=None, dtype=str)
            to_generalize = df.loc[:, index]
            from_generalize = possible_generalizations.iloc[:,[0,level]]
            for row in to_generalize:
                for i, pattern in from_generalize.iterrows():
                    if  str(pattern.iloc[0]) == str(row):
                        to_generalize.replace(to_replace = str(row), inplace=True, value = str(pattern.iloc[1]))
            df[index] = to_generalize
        
    print(df) 

def generalize_data(df:DataFrame, generalization_level:dict):
    for index, level in generalization_level.items():
        all_gen = create_generalization_hierarchies(index)
        to_generalize = df.loc[:, index]
        lookup = dict(zip(all_gen["0"], all_gen[str(level)]))
        for row in to_generalize:
            for original, anonymized in lookup.items():
                if str(original) == str(row):
                    to_generalize.replace(to_replace = str(row), inplace=True, value = str(anonymized))
        df[index] = to_generalize
        
    print(df)

generalize_data(df, gen)
'''graph.print_graph()'''
'''graph_generation(counter, graph)'''



