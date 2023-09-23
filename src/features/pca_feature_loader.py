from pandas import read_csv, DataFrame

class PCAFeatureLoader:

    features: DataFrame

    def __init__(self, path: str, **kwargs) -> None:
        self.features = read_csv(path, index_col='Year')
    
    def get_pca_features(
            self,
            ncomponents: int,
            window: int = 1,
            **kwargs
        ) -> DataFrame:
        
        features = self.features[self.features['window'] == window]
        result = DataFrame(
            features, 
            columns=[f'PCA{i}' for i in range(1, ncomponents + 1)],
            index=features.index
        )
        return result
