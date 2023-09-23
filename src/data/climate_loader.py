from pandas import DataFrame, read_csv
from src.data.utils import dat_to_dataframe
from src.common import get_justified_columns


class ClimateLoader:
    climate: DataFrame
    reanalysis: DataFrame
    
    def __init__(
            self,
            climate_path: str,
            reanalysis_path: str
        ) -> None:
        self.climate = read_csv(climate_path)
        self.reanalysis_path = read_csv(reanalysis_path, index_col='Date', parse_dates=True)

    @staticmethod
    def load_dat(temp_path: str, prec_path: str) -> DataFrame:

        temp = dat_to_dataframe(temp_path, True)
        prec = dat_to_dataframe(prec_path, True, 'Precipitation')

        return temp.join(prec)
    
    def __get_rolled_climate__(
            self,
            day_window: int = 14
        ) -> DataFrame:
        return (
            self
            .climate
            .set_index(['Year', 'Month', 'Day'])
            .rolling(day_window, 1, True)
            .mean()
            .reset_index()
        )
    
    def __get_pivoted_climate__(
            self,
            stat: str = 'Temperature',
            day_window: int = 14
        ) -> DataFrame:
        return (
            self
            .__get_rolled_climate__(day_window)
            .pivot(
                index='Year',
                columns=['Month', 'Day'],
                values=stat
            )
            .drop((2, 29), axis=1)
            .dropna()
        )
    
    def __get_rolled_pivoted_climate__(
            self,
            stat: str = 'Temperature',
            day_window: int = 14,
            year_window: int = 7
        ) -> DataFrame:
        return (
            self
            .__get_pivoted_climate__(stat, day_window)
            .rolling(year_window, 1, True)
            .mean()
        )
    
    def get_climate(
            self,
            stat: str = 'Temperature',
            day_window: int = 14,
            year_window: int = 7,
            p_threshold = 0.01
        ) -> DataFrame:

        climate = self.__get_pivoted_climate__(stat, day_window)
        climate_rolled = self.__get_rolled_pivoted_climate__(stat, day_window, year_window)
        days = get_justified_columns(climate, climate_rolled, p_threshold)

        return climate_rolled[days]
