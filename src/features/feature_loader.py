from pandas import DataFrame
from sklearn.decomposition import PCA
from typing import  Dict, List, Optional
from zhutils.tracheids import Tracheids

from src.common import get_justified_columns


class FeatureLoader:
    tracheids_name: str
    normalize_to: int
    trees_threshold: Optional[int]
    pca: Optional[PCA] = None

    def __init__(
            self,
            tracheids_name: str,
            tracheids_path: str,
            tracheids_trees: Optional[List[str]] = None,
            trees_threshold: int = 3,
            normalize_to: int = 15
        ) -> None:
        """
        Params:
            tracheids_name: Name for the Tracheids object
            tracheids_path: Path to the tracheids data file (.csv or .xlsx)
            tracheids_trees: List of tree names (required only for xlsx files)  
            trees_threshold: Minimum 
            normalize_to:
        """
        tracheids_trees = tracheids_trees or []

        tr = Tracheids(tracheids_name, tracheids_path, tracheids_trees)
        raw_norm_trs = tr.normalize(normalize_to)
        trees_count = raw_norm_trs[raw_norm_trs['№'] == 1].groupby('Year').count()
        years_to_save = trees_count[trees_count['Tree'] > trees_threshold].index
        raw_norm_trs = raw_norm_trs[raw_norm_trs['Year'].isin(years_to_save)]

        diams = (
            raw_norm_trs
            .pivot(
                index=['Tree', 'Year'],
                columns=['№'],
                values='Dmean'
            )
            .reset_index()
            .groupby('Year')
            .mean(numeric_only=True)
        )

        cwts = (
            raw_norm_trs
            .pivot(
                index=['Tree', 'Year'],
                columns=['№'],
                values='CWTmean'
            )
            .reset_index()
            .groupby('Year')
            .mean(numeric_only=True)
        )

        normalized_tracheids = (
            diams
            .add_prefix('D')
            .join(cwts.add_prefix('CWT'))
        )

        self.tracheids_name = tracheids_name
        self.normalize_to = normalize_to
        self.trees_threshold = trees_threshold
        self.__tracheids__ = tr
        self.__normalized_tracheids__ = normalized_tracheids
        self.__raw_normalized_tracheids__ = raw_norm_trs

    @property
    def features(self) -> DataFrame:
        """
        returns DataFrame with Year as index
        and N * 2 amount of columns named
        'D1', ... , 'DN', 'CWT1', ... , 'CWTN'
        where N == self.normalize_to
        """
        return self.__normalized_tracheids__
    

    def apply_zscore_to_features(self):
        from scipy.stats import zscore
        self.__normalized_tracheids__ = self.features.apply(zscore)
    
    def get_rolled_features(
            self,
            window: int,
            p_threshold: float = 0.01
        ) -> DataFrame:
        
        rolled_features = self.__normalized_tracheids__.rolling(window, 1, True).mean()
        justified_columns = get_justified_columns(
            self.__normalized_tracheids__,
            rolled_features,
            p_threshold
        )
        return rolled_features[justified_columns]
    
    def get_pca_features(
            self,
            ncomponents: int,
            window: int = 1,
            p_threshold: float = 0.01
        ) -> DataFrame:
        
        features = self.get_rolled_features(window, p_threshold)
        pca = PCA(ncomponents).fit(features)
        self.pca = pca
        pca_res = pca.transform(features)
        result = DataFrame(
            pca_res, 
            columns=[f'PCA{i}' for i in range(1, ncomponents + 1)],
            index=features.index
        )
        return result
    
    def get_not_normal_fractions(
            self,
            p_threshold: float = 0.05
        ) -> Dict[str, float]:
        """
        Returns dictionary like:
        {
            'Dmean': 0.05,
            'CWTmean': 0.1
        }
        Where keys are tracheid stats and values are fractions of not normaly
        distributed tracheid stats among all years
        """
        from scipy.stats import shapiro

        result = {}
        df = self.__raw_normalized_tracheids__

        for tr_stat in ['Dmean', 'CWTmean']:
            not_normal = 0
            total = 0
            for year in set(df['Year']):
                for i in range(1, self.normalize_to + 1):

                    total += 1
                    
                    df_ = df[(df['Year']==year) & (df['№']==i)]

                    if len(df_) < 3:
                        continue

                    _, p = shapiro(df_[tr_stat])

                    if p <= p_threshold:
                        not_normal += 1

            result[tr_stat] = not_normal / total
        
        return result
