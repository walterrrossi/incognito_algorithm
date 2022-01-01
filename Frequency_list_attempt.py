import pandas as pd
import copy
import time

from pandas.core.frame import DataFrame
from Graph import Graph
from Node import Node

start_time = time.time()
df = pd.read_csv("datasets/db_10000.csv", dtype=str)
df = df.drop(["id","disease"],1)

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
def get_frequency_list(df:DataFrame, qi_list:list, qi_frequency:dict = dict()):

    # qi_frequency: tuple -> (counts, {row_keys})

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

    qi_frequency = df[qi_list].value_counts()
    qi_frequency = qi_frequency.to_dict()

    counter = len(qi_frequency)
    return qi_frequency, counter


'''
    MAIN
    provo a trovare la frequency list
'''

# qi list
root_node_qi = ["age", "zip_code", "city_birth"]

# getting frequency list

[freq_list, counter] = get_frequency_list(df, root_node_qi)
#[freq_list, counter] = get_frequency_list_pandas(df, root_node_qi)

print(freq_list)
print("unique tuples: " + str(counter))
print("-----------------------------------------------------")

'''
    provo a trovare la frequency list per nodo non radice, utilizzo la vecchia frequency list
'''

#TODO: Creare il nuovo data frame composto con vecchia frequency list e generalization dimension table

'''
    For example, consider F1, the relational representation of
    the frequency set of the Patients table from Figure 1 with
    respect to hBirthdate, Sex, Zipcodei. Recall that in SQL
    the frequency set is computed by a COUNT(*) query with
    Birthdate, Sex, Zipcode as the GROUP BY clause. The fre-
    quency set (F2) of Patients with respect to hBirthdate, Sex,
    Z1i can be produced by joining F1 with the Zipcode dimen-
    sion table, and issuing a SUM(count) query with Birthdate,
    Sex, Z1 as the GROUP BY clause.
'''

# join with pandas
# new_df = df1.set_index('key').join(df2.set_index('key'))

'''
new_df = df
node_qi = ["age", "zip_code", "city_birth"]
old_freq_list = freq_list

[freq_list, counter] = get_frequency_list(new_df, node_qi, old_freq_list)

print(freq_list)
print("unique tuples: " + str(counter))
print("-----------------------------------------------------")

'''