import pandas as pd
import copy
import time

from pandas.core.frame import DataFrame
from Graph import Graph
from Node import Node

start_time = time.time()
df = pd.read_csv("datasets/db_100.csv", dtype=str)
df = df.drop(["id","disease"],1)

q_identifiers_list = [1, 2, 3]
generalization_levels = [4, 4, 6]
q_identifiers_dict = dict(zip(q_identifiers_list, generalization_levels))
q_identifiers_tag_id_dict = {"age" : 1, "city_birth" : 2 , "zip_code" : 3}

print("---------------------------------------------------")
print(df)
print("---------------------------------------------------")


'''
    FREQUENCY LIST CON ALGORITMO
    Questa funzione permette di ottenere la frequency list del dataset rispetto agli attributi qi indicati.
    
    :param dataframe contenente la tabella
    :param lista di qi da considerare nel conteggio per la frequency list
    
    :add_param frequency list da cui partire [default = empty]

    :return frequency_list and number of unique elements
'''
def get_frequency_list(df:DataFrame, qi_list:list):

    # qi_frequency: tuple -> (counts, {row_keys})
    qi_frequency = dict()

    for i, row in df.iterrows():
                
        # i = index row
        # row value

        qi_sequence = list()

        for qi in qi_list:
            qi_sequence.append(row[qi])

        # Skip if this row must be ignored:
        if qi_sequence is None:
            continue
        else:
            qi_sequence = tuple(qi_sequence)

        if qi_sequence in qi_frequency:
            occurrences = qi_frequency[qi_sequence][0] + 1          # add occurence of qi_sequence
            rows_set = qi_frequency[qi_sequence][1].union([i])      # add new index
            qi_frequency[qi_sequence] = (occurrences, rows_set)     # update value
        else:

            # Initialize number of occurrences and set of row indices:
            qi_frequency[qi_sequence] = (1, set())
            qi_frequency[qi_sequence][1].add(i)

    counter =  len(qi_frequency)   
    return qi_frequency, counter

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

    qi_frequency = df[qi_list].value_counts().rename_axis(qi_list).reset_index(name='counts')
    #qi_frequency = qi_frequency.to_dict()
    #qi_frequency = qi_frequency.to_frame()

    counter = len(qi_frequency)
    return qi_frequency, counter

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

'''
    MAIN
    provo a trovare la frequency list
'''

# qi list
root_node_qi = ["age", "zip_code", "city_birth"]

'''
start_time = time.time()
[freq_list, counter] = get_frequency_list(df, root_node_qi)
print("--- Execution time: "+str(time.time() - start_time)+"s")
'''

start_time = time.time()
[freq_list, counter] = get_frequency_list_pandas(df, root_node_qi)
print("--- Execution time: "+str(time.time() - start_time)+"s")

print(freq_list)
print("\n")
print("unique tuples: " + str(counter))
print("-----------------------------------------------------")

'''
    Ottengo le gerarchie di generalizzazione
'''

gen = {"zip_code" : 3, "age" : 2, "city_birth" : 2}
gen_hierarchies = create_generalization_hierarchies(gen)

parent_freq_list = freq_list

print(gen_hierarchies)

print("-----------------------------------------------------")

'''
    Produzione della lista delle chiavi
'''

#gen_attributes = list()
#gen_attributes_col = list()
final_attributes = list()

for tag in gen.keys():
    #gen_attributes.append(tag)
    #gen_attributes_col.append(str("{}{}").format(tag,"0"))
    final_attributes.append(str("{}{}").format(tag,gen[tag]))

#print(gen_attributes)
#print(gen_attributes_col)
print(final_attributes)

print("-----------------------------------------------------")

'''
    Join dei dataframe
'''

#new_table = parent_freq_list.set_index(gen_attributes).join(gen_hierarchies.set_index(gen_attributes_col))
#new_table = parent_freq_list.join(gen_hierarchies.set_index(gen_attributes_col), on=gen_attributes)
new_table = parent_freq_list.join(gen_hierarchies, lsuffix='', rsuffix='0')

# 0 magari va sostituito

print(new_table)

print("-----------------------------------------------------")

'''
    Ottengo Frequency list successiva
'''

# TODO: Assicurarsi che copia per riferimento non dia problemi oppure copiare il df per non lavorare sull originale

old_frequency_list_joined = new_table
node_qi = final_attributes

start_time = time.time()
[freq_list, counter] = get_frequency_list_pandas(old_frequency_list_joined, node_qi)
print("--- Execution time: "+str(time.time() - start_time)+"s")

print(freq_list)
print("\n")
print("unique tuples: " + str(counter))
print("-----------------------------------------------------")


'''
    OSSERVAZIONE: 

    Con 100 righe: 2a frequency list ci ~uguale alla 1a

    1 - 0.00598454475402832s
    2 - 0.0059871673583984375s

    Con 10000 righe: 2a frequency list più veloce della 1a

    1 - 0.03390932083129883s
    2 - 0.01795220375061035s

'''