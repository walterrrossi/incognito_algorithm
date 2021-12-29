import pandas as pd
import copy
from Graph import Graph

df = pd.read_csv("db_100.csv")
df = df.drop(["id","disease"],1)
print(df)

qi_list = ["a"]
gen = dict()
gen["a"] = 5
gen["b"] = 2
gen["vc"] = 3
K_anonimity = 2

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