import sqlite3

from src.data.climate_loader import ClimateLoader
from src.features import PCAFeatureLoader
from src.parameters import grid_search_pca


def train_tracheid_models(
        feature_loader_kwargs: dict,
        climate_loader_kwargs: dict,
        gridsearch_kwargs: dict,
        path_to_db: str,
    ) -> None:
    fl = PCAFeatureLoader(**feature_loader_kwargs)
    cl = ClimateLoader(**climate_loader_kwargs)

    db = sqlite3.connect(path_to_db)
    grid_search_pca(db, cl, fl, **gridsearch_kwargs)
    db.close()


import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error

from scipy.stats import zscore
import statsmodels.api as sm

from zhutils.correlation import dropna_pearsonr


def train_isotope_model(
        df: pd.DataFrame,
        isotopes_to_use: list[str],
        stat: str,
        save_path: str
    ) -> None:

    loo = LeaveOneOut()
    
    X = df[isotopes_to_use].apply(zscore).to_numpy()
    Y = df[stat].to_numpy()

    train_stats = {
        'R': [],
        'R_p': [],
        'F': [],
        'F_p': [],
        'R2': [],
        'RMSE': []
    }

    test_predictions = []
    coefs = []

    for train_index, test_index in loo.split(X, Y):
        reg = LinearRegression().fit(X[train_index], Y[train_index])
        train_prediction = reg.predict(X[train_index])
        test_prediction = reg.predict(X[test_index])[0]

        X_to_sm = sm.add_constant(X[train_index])
        model = sm.OLS(Y[train_index], X_to_sm).fit()
        f_value = model.fvalue
        p_value = model.f_pvalue

        train_stats['F'].append(f_value)
        train_stats['F_p'].append(p_value)

        r, p = dropna_pearsonr(Y[train_index], train_prediction)

        train_stats['R'].append(r)
        train_stats['R_p'].append(p)

        train_stats['R2'].append(r2_score(Y[train_index], train_prediction))
        mse = mean_squared_error(Y[train_index], train_prediction)
        train_stats['RMSE'].append(mse**.5)
        test_predictions.append(test_prediction)
        coefs.append(list(reg.coef_) + [reg.intercept_])


    final_coeffs = np.mean(coefs, axis=0)

    final_train_stats = {k:{} for k in train_stats}
    for k in train_stats:
        final_train_stats[k]['Mean'] = np.mean(train_stats[k])
        final_train_stats[k]['Std']  = np.std(train_stats[k])
        final_train_stats[k]['str'] = f"{final_train_stats[k]['Mean']:.2f}±{final_train_stats[k]['Std']:.2f}"

    final_test_stats = {}
    final_test_stats['R2'] = r2_score(Y, test_predictions)
    final_test_stats['R'], final_test_stats['R_p'] = dropna_pearsonr(Y, test_predictions)
    final_test_stats['RMSE'] = mean_squared_error(Y, test_predictions, squared=False)
        
    to_save = {'Year': list(df.index)}
    to_save.update({ k: df[k] for k in isotopes_to_use })
    to_save.update({
        'Real': Y,
        'Predicted_test': test_predictions
    })
    to_save = pd.DataFrame(to_save)
    to_save['Predicted_model'] = (
        to_save[isotopes_to_use].apply(zscore) * final_coeffs[:-1]
    ).sum(axis=1) + final_coeffs[-1]
    

    with pd.ExcelWriter(save_path) as writer:
        to_save.to_excel(writer, sheet_name="data", index=False)
        pd.DataFrame(
            final_train_stats
        ).to_excel(writer, sheet_name="train_stats")
        pd.DataFrame(
            final_test_stats,
            index=[0]
        ).to_excel(writer, sheet_name="test_stats", index=False)
        pd.DataFrame(
            {k:v for k, v in zip(isotopes_to_use+['Const'], final_coeffs)},
            index=[0]
        ).to_excel(writer, sheet_name="coeffs", index=False)

