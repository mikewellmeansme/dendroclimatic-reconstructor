from pandas import DataFrame
from typing import (
    Iterable,
    Tuple,
    Union,
    Callable,
    Dict,
    List,
    Any
)
from sklearn.linear_model import LinearRegression
from numpy import array, mean, std


class RollingLeaveOneOut:
    models: Iterable
    folds: List
    real_y: array
    predictions_train: List
    predictions_test: array

    def __init__(
            self,
            df: DataFrame,
            window: int,
            x_cols: Iterable,
            y_col: Union[str, Tuple, int],
            reg_cls = LinearRegression
        ) -> None:
        
        models = []
        folds = []
        predictions_train = []
        predictions_test = []
        
        for i in df.index:
            df_ = df.drop(set(range(i-window//2, i+window//2)).intersection(df.index))
            model = reg_cls().fit(df_[x_cols].to_numpy(), df_[y_col].to_numpy())
            models.append(model)
            folds.append(df_[y_col])
            predictions_test.append(model.predict([df[x_cols].loc[i].to_numpy()])[0])
            predictions_train.append(model.predict(df_[x_cols].to_numpy()))
        
        self.models = models
        self.folds = folds
        self.real_y = list(df[y_col])
        self.predictions_train = predictions_train
        self.predictions_test = predictions_test

    def get_metrics(
            self,
            metrics: Dict[str, Callable[[Iterable, Iterable], Any]]
        ) -> Dict[str, Any]:
        result = {}
        for name, metric in metrics.items():
            result[f'{name}_test'] = metric(self.real_y, self.predictions_test)
            train_metrics = [
                metric(fold, pred) 
                for fold, pred
                in zip(self.folds, self.predictions_train)
            ]
            result[f'{name}_train'] = mean(train_metrics)
            result[f'{name}_train_std'] = std(train_metrics)
        
        return result
    
    def get_coeefs(self, full=False) -> Union[Iterable[float], Iterable[Iterable[float]]]:
        if not isinstance(self.models[0], LinearRegression):
            raise NotImplementedError('Only LinearRegression is supported!')
        
        result = []
        
        for model in self.models:
            result.append(list(model.coef_) + [model.intercept_])
        
        if full:
            return result
        else:
            return mean(result, axis=0)
