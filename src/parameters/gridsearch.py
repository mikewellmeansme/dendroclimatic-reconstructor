import sqlite3
import concurrent.futures
import time

from sklearn.metrics import r2_score, mean_squared_error
from typing import Optional, Union

from src.features import FeatureLoader, PCAFeatureLoader
from src.data import ClimateLoader
from src.validation import RollingLeaveOneOut
from src.visualization import progressBar



def worker(
        column,
        df,
        year_window,
        pca_features_columns
    ):
    rloo = RollingLeaveOneOut(df, year_window, pca_features_columns, column)
    res = rloo.get_metrics({
        'R2': r2_score,
        'MSE': mean_squared_error
    })

    month, day = column

    json_to_add = str({
        "coeffs": list(rloo.get_coeefs()),
        "predictions_test": list(rloo.predictions_test),
        "real_y": list(rloo.real_y),
    }).replace("'", '"')

    return f"""
        {month}, {day},
        {res['R2_test']}, {res['R2_train']}, {res['R2_train_std']},
        {res['MSE_test']}, {res['MSE_train']}, {res['MSE_train_std']},
        '{json_to_add}'
    """

def grid_search_pca(
        db: sqlite3.Connection,
        cl: ClimateLoader,
        fl: Union[FeatureLoader, PCAFeatureLoader],
        year_window_range: Optional[range] = None,
        pca_ncomponents_range: Optional[range] = None,
        day_window: int = 1,
        p_threshold: float = 0.001,
        kind_name: str = 'REAL',
        table_name: str = 'results',
        max_workers: int = 6,
        stat: str = "Temperature"
    ):
    cursor = db.cursor()
    year_window_range = year_window_range or range(1, 15)
    pca_ncomponents_range = pca_ncomponents_range or range(2, 10)


    for year_window in progressBar(year_window_range):
        start_time_1 = time.perf_counter()
        clim_ = cl.get_climate(stat, day_window, year_window, p_threshold)
        end_time_1 = time.perf_counter()

        print(f'Climate creation {year_window}:', end_time_1-start_time_1)

        for pca_ncomponents in pca_ncomponents_range:
            start_time_2 = time.perf_counter()

            pca_features = fl.get_pca_features(pca_ncomponents, year_window)
            df = pca_features.join(clim_, how='inner')

            end_time_2 = time.perf_counter()
            print(f'PCA creation {pca_ncomponents}:', end_time_2-start_time_2)
            

            start_time_3 = time.perf_counter()
            futures = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                for column in clim_.columns:
                    futures.append(executor.submit(
                        worker,
                        column,
                        df.copy(),
                        year_window,
                        pca_features.columns
                    ))

            concurrent.futures.wait(futures)

            end_time_3 = time.perf_counter()
            print('Futures end:', end_time_3-start_time_3)

            start_time_4 = time.perf_counter()
            prefix = f"'{kind_name}', {day_window}, {year_window}, {pca_ncomponents}, '{stat}', "
            values_to_insert = [f"({prefix} {future.result()})" for future in futures]
            cursor.execute(f"""INSERT INTO {table_name} VALUES {", ".join(values_to_insert)};""")
            end_time_4 = time.perf_counter()

            print('DB writing:', end_time_4-start_time_4)
            
    db.commit()
    
