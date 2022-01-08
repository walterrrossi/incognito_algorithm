import pandas as pd
import copy
from pandas.core.frame import DataFrame

# -------------------------------------------------------------------


def create_generalization_hierarchies(generalizations_tag: list, q_identifiers_tag_id_dict: dict, q_identifiers_id_lev_dict: dict):
    """
    Questa funziona genera una tabella con tutte le possibili generalizzazioni per ogni QI, crea un dizionario di dizionari
    con come chiave il tag del QI, mentre come valore un dizionario che a sua volta ha come chiave il livello di generalizzazione e come 
    valore la lista di tutte le generalizzazioni di quel livello

    Args:
        generalizations_tag (list): lista contenente il i tag dei vari QI da generalizzare
        q_identifiers_tag_id_dict (dict): dizionario contenente coppie chiavi-valore del tipo (tag del QI):(identificativo intero usato nei nodi)
        q_identifiers_id_lev_dict (dict): dizionario contenente coppie chiavi-valore del tipo (identificativo intero usato nei nodi):(livello di generalizzazione relativo a quel QI)

    Returns:
        [DataFrame]: DataFrame contenente tutte le generalizzazioni, una per ogni colonna che ha come indice il tag del QI e il relativo livello di generalizzazione
    """
    all_gen = pd.DataFrame()
    for tag in generalizations_tag:
        # path = str("datasets/{}_generalization.csv").format(str(tag))
        path = str(
            "datasets/adult/hierarchies/adult_hierarchy_{}.csv").format(str(tag))
        df_one_gen = pd.read_csv(path, header=None, sep=(";"), dtype=str)
        for key, qi_id in q_identifiers_tag_id_dict.items():
            if key == tag:
                for i in range(0, q_identifiers_id_lev_dict[qi_id]):
                    tmp_df = pd.DataFrame()
                    column_name = str("{}|{}").format(key, i)
                    tmp_df[column_name] = df_one_gen.iloc[:, i]
                    all_gen = pd.concat([all_gen, tmp_df], axis=1)

    return all_gen

# -------------------------------------------------------------------


def generalize_data(dataset: DataFrame, generalization_levels: dict, all_generalizations: DataFrame):
    """
    Questa funzione prende in input il df, le generalizzazioni richieste e un dizionario contenente tutte le generalizzazioni per ogni QI.
    Cicla su tutte le coppie chiavi-valore presenti nel dizionario delle generalizzazioni richieste e in base al livello di generalizzazione
    richiesto prende dalla tabella contenente tutte le gen. la prima colonna(valore originale) e la colonna del livello richiesto.
    A questo punto sostituisce con il valore anonimizzato

    Args:
        dataset (DataFrame): DataFrame originale contenente tutte le tuple da generalizzare
        generalization_levels (dict): un dizionario contenente coppie di chiavi-valore del tipo (id intero del QI):(livello di generalizzazione),
        che identificano i livelli al quale è necessario generalizzare i vari QI
        all_generalizations (DataFrame): DataFrame che ritorna la funziona create_generalization_hierarchies (vedi sopra)

    Returns:
        [DataFrame]: DataFrame contente solo le colonne di QI generalizzati al livello richiesto
    """
    df_original = copy.copy(dataset)
    df_generalized = pd.DataFrame()
    for index, level in generalization_levels.items():
        to_generalize = df_original.loc[:, index]
        lev_string = str("{}|{}").format(index, level)
        ind_string = str("{}|{}").format(index, 0)
        lookup = pd.Series(
            all_generalizations[lev_string].values, index=all_generalizations[ind_string]).to_dict()
        for row in to_generalize:
            for original, anonymized in lookup.items():
                if str(original) == str(row):
                    to_generalize.replace(to_replace=str(
                        original), inplace=True, value=str(anonymized))
        # THIS COMMENTED VERSION IS TO HAVE COLUMNS ORIGINAL NAMES
        # df_generalized[index] = to_generalize
        df_generalized[lev_string] = to_generalize

    return df_generalized
