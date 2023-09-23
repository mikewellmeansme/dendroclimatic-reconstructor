from pandas import DataFrame
from typing import List, Tuple
from zhutils.correlation import dropna_pearsonr


def get_justified_columns(
        df: DataFrame,
        rolled_df: DataFrame,
        p_threshold = 0.01,
    ) -> List[Tuple[int, int]]:
    """
    Returns the list of columns, which have
    significant (p<p_threshold) correlation 
    with their rolling versions
    """
    
    columns_to_use = []

    for column in df:
        r, p = dropna_pearsonr(
            df[column],
            rolled_df[column]
        )
        if p < p_threshold:
            columns_to_use.append(column)
    
    return columns_to_use
