import pandas as pd
import copy
from Graph import Graph
from Node import Node
from dgh import CsvDGH
import argparse

df = pd.read_csv("db_100.csv")
df = df.drop(["id","disease"],1)
print(df)
counter = 1




qi_list = ["a"]
gen = dict()
gen["a"] = 5
gen["b"] = 2
gen["vc"] = 3
K_anonimity = 2
graph = Graph()
a = Node(counter, False, [1, 2], [1,0])
counter+=1
b = Node(counter, False, [1, 2], [1,1])
counter+=1
c = Node(counter, False, [1, 2], [0,2])
counter+=1
d = Node(counter, False, [1, 2], [1,2])
counter+=1
e = Node(counter, False, [3, 2], [1,0])
counter+=1
f = Node(counter, False, [3, 2], [1,1])
counter+=1
g = Node(counter, False, [3, 2], [0,2])
counter+=1
h = Node(counter, False, [3, 2], [1,2])
counter+=1
i = Node(counter, False, [3, 1], [1,0])
counter+=1
l = Node(counter, False, [3, 1], [0,1])
counter+=1
m= Node(counter, False, [3, 1], [1,1])
counter+=1



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

graph_generation(counter, graph)   
   
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




counter=1

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
                
               
                nodeTemp = Node(counter,False, [*qi1, qi2[-1]], [*gl1, gl2[-1]])
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
    
    
    newGraph.print_nodes()
    

    unique_result_edges = []
    
    for e in candidate_edges:
        if e not in unique_result_edges:
            unique_result_edges.append(e)

    print(unique_result_edges)

    edges = []

    edges_to_remove=[]
    
    for d1 in range(len(unique_result_edges)):
        for d2 in range(len(unique_result_edges)): 
            print(str(d1) + str(d2))
            if unique_result_edges[d1][1]==unique_result_edges[d2][0]:
                edges_to_remove.append([unique_result_edges[d1][0],unique_result_edges[d2][1]])
    
    final_edges = []
    for e in unique_result_edges:
        if e not in edges_to_remove:
            final_edges.append(e)
    
    print(unique_result_edges)
    print(final_edges)

    newGraph.edges=final_edges

    newGraph.print_graph()