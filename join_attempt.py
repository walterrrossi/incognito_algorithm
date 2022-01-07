from numpy import pi
import pandas as pd
import copy
import time
import collections

from pandas.core.frame import DataFrame

df = pd.read_csv("datasets/db_100.csv", dtype=str)
df = df.drop(["id","disease"],1)


def create_generalization_hierarchies(generalization_level: list, q_identifiers_tag_id_dict: dict, q_identifiers_id_lev_dict: dict):
    all_gen = pd.DataFrame()
    for tag in generalization_level:
        #path = str("datasets/{}_generalization.csv").format(str(tag))
        path = str("datasets/{}_generalization.csv").format(str(tag))
        df_one_gen = pd.read_csv(path, header=None, dtype=str)
        for key, qi_id in q_identifiers_tag_id_dict.items():
            if key == tag:
                for i in range(0, q_identifiers_id_lev_dict[qi_id]):
                    tmp_df = pd.DataFrame()
                    column_name = str("{}{}").format(key, i)
                    tmp_df[column_name] = df_one_gen.iloc[:, i]
                    all_gen = pd.concat([all_gen, tmp_df], axis=1)

    return all_gen

def generalize_data(dataset: DataFrame, generalization_levels: dict, all_generalizations: DataFrame):
    df_original = copy.copy(dataset)
    df_generalized = pd.DataFrame()
    for index, level in generalization_levels.items():
        to_generalize = df_original.loc[:, index]
        lev_string = str("{}{}").format(index, level)
        ind_string = str("{}{}").format(index, 0)
        lookup = pd.Series(
            all_generalizations[lev_string].values, index=all_generalizations[ind_string]).to_dict()
        for row in to_generalize:
            for original, anonymized in lookup.items():
                if str(original) == str(row):
                    to_generalize.replace(to_replace=str(
                        original), inplace=True, value=str(anonymized))
        df_generalized[lev_string] = to_generalize

    return df_generalized

def get_frequency_list_pandas(df: DataFrame, qi_list: list):

    # tuple -> (counts, {row_keys})
    qi_frequency = dict()
    qi_list = list(df.columns)

    if 'counts' in df.columns:
        qi_list.remove("counts")
        df = df.rename(columns={"counts": "old_counts"})

    qi_frequency = df.value_counts().reset_index(name='counts')
    #qi_frequency = df[qi_list].value_counts().rename_axis(qi_list).reset_index(name='counts')
    #qi_frequency = qi_frequency.to_dict()
    #qi_frequency = qi_frequency.to_frame()

    if 'old_counts' in qi_frequency.columns:
        print("--- NEW FREQ SET 1 ---")
        print("Faccio il count dei gruppi che ho con i nuovi attributi (count è quello nuovo)")
        print(qi_frequency)
        new_counts = qi_frequency["old_counts"] * qi_frequency["counts"]
        qi_frequency = qi_frequency.drop(['old_counts', 'counts'], axis=1)
        qi_frequency["counts"] = new_counts
        print("--- NEW FREQ SET 2---")
        print("Moltiplico i conteggi per trovare il count di ogni gruppo")
        print(qi_frequency)
        aggregation_functions = {'counts': 'sum'}
        qi_frequency = qi_frequency.groupby(qi_list).aggregate(aggregation_functions).reset_index()
        print("--- NEW FREQ SET 3---")
        print("Sommo le tuple uguali sommando i counter")
        print(qi_frequency)

    return qi_frequency


q_identifiers_list = [1, 2, 3]
generalization_levels = [4, 4, 6]
q_identifiers_tag_id_dict = {"age" : 1, "city_birth" : 2 , "zip_code" : 3}
q_identifiers_list_string = ["age", "city_birth", "zip_code"]
q_identifiers_id_lev_dict = dict(zip(q_identifiers_list, generalization_levels))

dataset = df;
generalizations_table = create_generalization_hierarchies(q_identifiers_list_string, q_identifiers_tag_id_dict, q_identifiers_id_lev_dict)

print("*****************************************************************************************")
print("*****************************************************************************************")
print("*****************************************************************************************")
qi_dict_node = {'zip_code': 0}
print(qi_dict_node)

root_T = generalize_data(dataset, qi_dict_node, generalizations_table)

print("--- ROOT T ---")
print(root_T)

root_T['counts'] = 1
print(root_T)
aggregation_functions = {'counts': 'sum'}
root_frequency_set1 = root_T.groupby("zip_code0").aggregate(aggregation_functions).reset_index()

print("--- ROOT FREQ SET ---")
print(root_frequency_set1)

root_T = generalize_data(dataset, qi_dict_node, generalizations_table)

root_frequency_set = get_frequency_list_pandas(root_T.filter(items=["zip_code0"]), [])

print("--- ROOT FREQ SET 2 ---")
print(root_frequency_set)

print("--------------------------------------------------------------------------------------")

qi_dict_node = {'zip_code': 1}
print(qi_dict_node)

qi_with_levels = []
for key in qi_dict_node.keys():
    qi_with_levels.append(str("{}{}").format(key,str(qi_dict_node[key])))   

print("qi with levels")
print(qi_with_levels)    

print("--- OLD FRQ SET ---")
print("Frequency set di parenza del nodo padre")
print(root_frequency_set)

gen_table_node = generalizations_table.filter(items=["zip_code0","zip_code1"]).dropna()

print("--- GEN TABLE ---")
print("tabella delle generalizzazioni")
print(gen_table_node)

joined_table = pd.merge(root_frequency_set, gen_table_node, on='zip_code0').filter(items=["zip_code1","counts"])

print("--- JOINED TABLE ---")
print("join di frequency set vecchio e tabella delle generalizzazioni")
print(joined_table)

aggregation_functions = {'counts': 'sum'}
trial = joined_table.groupby("zip_code1").aggregate(aggregation_functions)

print("--- TRIAL COUNT TABLE ---")
print("Uso aggregate per il count ?")
print(trial)

new_frequency_set = get_frequency_list_pandas(joined_table, [])

print("--- FINAL FREQ SET ---")
print(" => frequency set finale con nuovo metodo")
print(new_frequency_set)

gen_T_try = generalize_data(dataset, qi_dict_node, generalizations_table)
fs_try = get_frequency_list_pandas(gen_T_try, [])

print("--- ALTERNATIVE FREQ SET ---")
print(" => frequency set finale con vecchio metodo")
print(fs_try)

print("*****************************************************************************************")
print("*****************************************************************************************")
print("*****************************************************************************************")
qi_dict_node = {'age': 0, 'zip_code': 0}
print(qi_dict_node)

root_T = generalize_data(dataset, qi_dict_node, generalizations_table)

print("--- ROOT T ---")
print(root_T)

root_frequency_set = get_frequency_list_pandas(root_T, [])

print("--- NEW FREQ SET ---")
print(root_frequency_set)

print("--------------------------------------------------------------------------------------")

qi_dict_node = {'age': 1, 'zip_code': 0}
print(qi_dict_node)

qi_with_levels = []
for key in qi_dict_node.keys():
    qi_with_levels.append(str("{}{}").format(key,str(qi_dict_node[key])))   

print("qi with levels")
print(qi_with_levels)    

print("--- OLD FRQ SET ---")
print(root_frequency_set)

gen_table_node = generalizations_table.filter(items=["age0","age1"]).dropna()

print("--- GEN TABLE ---")
print(gen_table_node)

#joined_table = pd.merge(left=root_frequency_set, right=generalizations_table, left_on=['age0'], right_on=['age0'])
#joined_table = generalizations_table.join(root_frequency_set, on='age0', how='inner')
joined_table = pd.merge(root_frequency_set, gen_table_node, on='age0').filter(items=["age1","zip_code0","counts"])
print("--- JOINED TABLE ---")
print(joined_table)

aggregation_functions = {'counts': 'sum'}
trial = joined_table.groupby(["age1","zip_code0"]).aggregate(aggregation_functions)

print("--- TRIAL COUNT TABLE ---")
print("Uso aggregate per il count ?")
print(trial)
print(trial.iloc[[1]])

frequency_set = get_frequency_list_pandas(joined_table, [])

print("--- NEW FREQ SET ---")
print(frequency_set)


##
#
# 10000 -> 1864 s -> 123, 312
#
##