from datetime import datetime
from pandas import DataFrame


def dat_to_dataframe(path, date_together=False, stat='Temperature'):
    with open(path, 'r') as f:
        dates = []
        vals = []
        for line in f:
            if line.startswith('#'):
                continue
            else:
                if date_together:
                    date, val = line.strip().split()
                    year = int(date[:4])
                    month = int(date[4:6])
                    day = int(date[6:])
                else:
                    year, month, day, val = line.strip().split()
                    year, month, day = int(year), int(month), int(day)
                val = float(val)
                date = datetime(year, month, day)
                dates.append(date)
                vals.append(val)
        df = DataFrame({'Date': dates, stat: vals})
    
    return df.set_index('Date')
