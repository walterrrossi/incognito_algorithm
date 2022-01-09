import pandas as pd
import copy
from pandas.core.frame import DataFrame

# -------------------------------------------------------------------


def create_generalization_hierarchies(generalizations_tag: list, q_identifiers_tag_id_dict: dict, q_identifiers_id_lev_dict: dict):
    """
    This function generates a table with all the possible generalizations for each QI, creates a dictionary
    with as key the tag of the QI, while as value a dictionary that has as key the level of generalization and as 
    value the list of all generalizations of that level.

    Args:
        generalizations_tag (list): list containing the tags of the various QIs to be generalized
        q_identifiers_tag_id_dict (dict): dictionary containing key-value pairs of type (QI names):(integer identifier used in nodes)
        q_identifiers_id_lev_dict (dict): dictionary containing key-value pairs of the type (integer identifier used in the nodes):(level of generalization relative to that QI)

    Returns:
        [DataFrame]: DataFrame containing all generalizations, one for each column that has as index the QI name and its level of generalization
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
    This function takes as input the df, the required generalizations, and a dictionary containing all generalizations for each QI.
    It cycles through all the key-value pairs in the dictionary of the required generalizations and according to the level of generalization
    it takes from the table containing all the generalizations the first column (original value) and the column of the requested level.
    At this point it replaces with the anonymized value.

    Args:
        dataset (DataFrame): original DataFrame containing all tuples to be generalized
        generalization_levels (dict): a dictionary containing key-value pairs of the type (integer id of the QI):(level of generalization),
                                        that identify the levels to which it is necessary to generalize the various QIs
        all_generalizations (DataFrame): DataFrame che ritorna la funziona create_generalization_hierarchies (vedi sopra)

    Returns:
        [DataFrame]: DataFrame containing only QI columns generalized to the required level
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
        df_generalized[lev_string] = to_generalize

    return df_generalized
